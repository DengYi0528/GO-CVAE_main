import anndata
import keras
import tensorflow as tf
import pandas as pd

from keras.callbacks import EarlyStopping, History, ReduceLROnPlateau, LambdaCallback
from keras.engine.saving import model_from_json
from keras.layers import Dense, BatchNormalization, Dropout, Lambda, Input, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.utils import to_categorical
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Embedding, Flatten
from keras import backend as K

from GOCVAE.models._activations import ACTIVATIONS
from GOCVAE.models._layers import LAYERS, ConditionEmbeddingLayer
from GOCVAE.models._losses import LOSSES
from GOCVAE.models._utils import print_progress, sample_z
from GOCVAE.utils import *



class GOCVAE(object):

    def __init__(self, gene_size: int, conditions: list, n_topic=10, **kwargs):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.6

        session = tf.Session(config=config)
        K.set_session(session)

        self.gene_size = gene_size
        self.n_topic = n_topic

        self.conditions = sorted(conditions)
        self.n_conditions = len(self.conditions)

        self.lr = kwargs.get("learning_rate", 0.001)
        self.alpha = kwargs.get("alpha", 0.0001)
        self.eta = kwargs.get("eta", 50.0)
        self.dr_rate = kwargs.get("dropout_rate", 0.1)
        self.model_path = kwargs.get("model_path", "./models/GOCVAE/")
        self.loss_fn = kwargs.get("loss_fn", 'mse')
        self.ridge = kwargs.get('ridge', 0.1)
        self.scale_factor = kwargs.get("scale_factor", 1.0)
        self.clip_value = kwargs.get('clip_value', 3.0)
        self.epsilon = kwargs.get('epsilon', 0.01)
        self.output_activation = kwargs.get("output_activation", 'linear')
        self.use_batchnorm = kwargs.get("use_batchnorm", True)

        self.architecture = kwargs.get("architecture", [128, 128])
        self.size_factor_key = kwargs.get("size_factor_key", 'size_factors')
        self.device = kwargs.get("device", "gpu") if len(K.tensorflow_backend._get_available_gpus()) > 0 else 'cpu'

        self.gene_names = kwargs.get("gene_names", None)
        self.model_name = kwargs.get("model_name", "GOCVAE")
        self.class_name = kwargs.get("class_name", 'GOCVAE')

        self.ctrl_expr = Input(shape=(self.gene_size,), name="ctrl_expr")
        self.pert_expr_true = Input(shape=(self.gene_size,), name="pert_expr_true")
        self.size_factor = Input(shape=(1,), name='size_factor')
        self.x_hat = tf.random.normal([1, gene_size])
        self.z = Input(shape=(self.n_topic,), name="latent_data")

        self.use_gears_embedding = kwargs.get("use_gears_embedding", True)
        self.gears_embedding_dim = kwargs.get("gears_embedding_dim", 64)
        self.num_go_gnn_layers = kwargs.get("num_go_gnn_layers", 1)
        self.go_graph_path = kwargs.get("go_graph_path", None)

        self.encoder_labels = Input(shape=(1,), dtype='int32', name="encoder_labels")
        self.decoder_labels = Input(shape=(1,), dtype='int32', name="decoder_labels")

        self.condition_embedding_layer = None
        self.unique_genes = None
        self.gene2idx = None
        self.go_adj_matrix = None

        if not self.use_gears_embedding:
            self.condition_embedding = Embedding(
                input_dim=self.n_conditions,
                output_dim=self.n_topic,
                embeddings_initializer='glorot_uniform',
                name='condition_embedding'
            )

        self.condition_encoder = kwargs.get("condition_encoder", None)
        self.ctrl_key = kwargs.get("ctrl_key", "control")
        self.pert_key = kwargs.get("pert_key", "perturbation")
        self.nperts_key = kwargs.get("nperts_key", "nperts")

        self.pawine_lambda = kwargs.get("pawine_lambda", 10.0)
        self.pawine_lambda_d = kwargs.get("pawine_lambda_d", 0.01)
        self.go_similarity_dict = {}
        self.data_path = kwargs.get("data_path", "../data/GEARS/")
        self.data_name = kwargs.get("data_name", "norman")
        self.go_similarity_threshold = kwargs.get("go_similarity_threshold", 0.15)
        self.smart_batch_config_path = kwargs.get("smart_batch_config_path", None)
        self.pawine_triplets = None

        self.gears_gamma = kwargs.get("gears_gamma", 2)
        self.gears_direction_lambda = kwargs.get("gears_direction_lambda", 1e-3)
        self.gears_lambda = kwargs.get("gears_lambda", 1.0)

        self.disp = tf.Variable(
            initial_value=np.ones(self.gene_size),
            dtype=tf.float32,
            name="disp"
        )

        self.network_kwargs = {
            "gene_size": self.gene_size,
            "n_topic": self.n_topic,
            "conditions": self.conditions,
            "dropout_rate": self.dr_rate,
            "loss_fn": self.loss_fn,
            "output_activation": self.output_activation,
            "size_factor_key": self.size_factor_key,
            "architecture": self.architecture,
            "use_batchnorm": self.use_batchnorm,
            "gene_names": self.gene_names,
            "condition_encoder": self.condition_encoder,
            "train_device": self.device,
        }

        self.training_kwargs = {
            "learning_rate": self.lr,
            "alpha": self.alpha,
            "eta": self.eta,
            "ridge": self.ridge,
            "scale_factor": self.scale_factor,
            "clip_value": self.clip_value,
            "model_path": self.model_path,
        }

        kwargs.update({"model_name": "cvae", "class_name": "GOCVAE"})


        self.init_w = keras.initializers.glorot_normal()

        if kwargs.get("construct_model", True):
            self.construct_network()

        if kwargs.get("construct_model", True) and kwargs.get("compile_model", True):
            self.compile_models()

        print_summary = kwargs.get("print_summary", False)
        if print_summary:
            self.encoder_model.summary()
            self.decoder_model.summary()
            self.cvae_model.summary()

    @classmethod
    def from_config(cls, config_path, new_params=None, compile=True, construct=True):

        import json

        with open(config_path, 'rb') as f:
            class_config = json.load(f)

        class_config['construct_model'] = construct
        class_config['compile_model'] = compile

        if new_params:
            class_config.update(new_params)

        return cls(**class_config)


    def _initialize_gears_embeddings(self):

        from GOCVAE.utils import (parse_any_pert, get_similarity_network,
                                 filter_genes_with_go_annotation, build_go_adjacency_matrix)
        from GOCVAE.models._layers import ConditionEmbeddingLayer
        import os
        import pandas as pd

        if not hasattr(self, 'gene_names') or self.gene_names is None:
            raise ValueError("The gene_names parameter was not provided during model initialization.")

        self.all_genes = self.gene_names
        self.num_all_genes = len(self.all_genes)
        self.gene2idx = {gene: idx for idx, gene in enumerate(self.all_genes)}

        condition_to_genes = {}
        pert_genes = set()

        for cond in self.conditions:
            genes = parse_any_pert(cond) or []
            condition_to_genes[cond] = genes
            pert_genes.update(genes)

        max_genes = max((len(g) for g in condition_to_genes.values()), default=0)

        missing = pert_genes - set(self.all_genes)
        if missing:
            print(f"  {len(missing)}perturbed genes are not in adata.var: {list(missing)[:5]}")

        if hasattr(self, 'filtered_genes_with_go') and self.filtered_genes_with_go:
            filtered_genes = [g for g in self.all_genes if g in self.filtered_genes_with_go]
        else:
            filtered_genes = filter_genes_with_go_annotation(self.all_genes, self.data_path)

        if len(filtered_genes) < 5:
            self.use_gears_embedding = False
            return

        if hasattr(self, 'go_df') and self.go_df is not None:
            go_df = self.go_df
            print("  Using in-memory GO graphs")

        elif self.go_graph_path and os.path.exists(self.go_graph_path):
            go_df = pd.read_csv(self.go_graph_path)

        else:
            default_path = os.path.join(self.model_path, 'go_graph_all_genes.csv')
            if self.go_graph_path is None and os.path.exists(default_path):
                go_df = pd.read_csv(default_path)
                self.go_graph_path = default_path
            else:
                print(" Building a new GO graph...")
                go_df = get_similarity_network(
                    k=20,
                    data_path=self.data_path,
                    data_name='auto_generated',
                    pert_list=filtered_genes
                )

                os.makedirs(self.model_path, exist_ok=True)
                save_path = os.path.join(self.model_path, 'go_graph_all_genes.csv')
                go_df.to_csv(save_path, index=False)
                self.go_graph_path = save_path

        self.go_adj_matrix = build_go_adjacency_matrix(
            go_df,
            self.all_genes,
            self.gene2idx
        )

        try:
            self.condition_embedding_layer = ConditionEmbeddingLayer(
                conditions=self.conditions,
                condition_to_genes=condition_to_genes,
                all_genes=self.all_genes,
                gene2idx=self.gene2idx,
                go_adj_matrix=self.go_adj_matrix,
                embedding_dim=self.gears_embedding_dim,
                num_gnn_layers=self.num_go_gnn_layers,
                max_genes_per_condition=max(5, max_genes + 1),
                name='gears_condition_embedding'
            )

            n_params = (
                    self.num_all_genes * self.gears_embedding_dim +
                    self.num_go_gnn_layers * self.gears_embedding_dim ** 2 +
                    3 * (self.gears_embedding_dim ** 2 + self.gears_embedding_dim)
            )

            print(f"\n GEARS Embedding Layer Created Successfully")

        except Exception as e:
            print(f"Creation failed: {e}")
            import traceback
            traceback.print_exc()
            self.use_gears_embedding = False
            return

    def _encoder(self, name="encoder"):

        if self.use_gears_embedding and self.condition_embedding_layer is None:
            self._initialize_gears_embeddings()

        if self.use_gears_embedding and self.condition_embedding_layer is not None:
            encoder_labels_emb = self.condition_embedding_layer(self.encoder_labels)
        else:
            encoder_labels_emb = self.condition_embedding(self.encoder_labels)
            encoder_labels_emb = Flatten()(encoder_labels_emb)

        h = concatenate([self.ctrl_expr, encoder_labels_emb], axis=1)

        for idx, n_neuron in enumerate(self.architecture):
            h = Dense(n_neuron, kernel_initializer=self.init_w, use_bias=False)(h)
            if self.use_batchnorm:
                h = BatchNormalization()(h)
            h = LeakyReLU()(h)
            if self.dr_rate > 0:
                h = Dropout(self.dr_rate)(h)

        mean = Dense(self.n_topic, kernel_initializer=self.init_w)(h)
        log_var = Dense(self.n_topic, kernel_initializer=self.init_w)(h)
        z = Lambda(sample_z, output_shape=(self.n_topic,))([mean, log_var])

        model = Model(inputs=[self.ctrl_expr, self.encoder_labels], outputs=[mean, log_var, z], name=name)

        return mean, log_var, model


    def _decoder(self, name="decoder"):
        if self.use_gears_embedding and self.condition_embedding_layer is not None:
            decoder_labels_emb = self.condition_embedding_layer(self.decoder_labels)
        else:
            decoder_labels_emb = self.condition_embedding(self.decoder_labels)
            decoder_labels_emb = Flatten()(decoder_labels_emb)

        h = concatenate([self.z, decoder_labels_emb], axis=1)

        for idx, n_neuron in enumerate(self.architecture[::-1]):
            h = Dense(n_neuron, kernel_initializer=self.init_w, use_bias=False)(h)
            if self.use_batchnorm:
                h = BatchNormalization()(h)
            h = LeakyReLU()(h)
            if self.dr_rate > 0:
                h = Dropout(self.dr_rate)(h)

        h = Dense(self.gene_size, activation=None,
                  kernel_initializer=self.init_w,
                  use_bias=True)(h)
        h = ACTIVATIONS[self.output_activation](h)

        model = Model(inputs=[self.z, self.decoder_labels], outputs=[h], name=name)

        return model

    def construct_network(self):
        self.mu, self.log_var, self.encoder_model = self._encoder(name="encoder")
        self.decoder_model = self._decoder(name="decoder")
        inputs = [self.ctrl_expr, self.encoder_labels, self.decoder_labels, self.pert_expr_true]
        self.z = self.encoder_model(inputs[:2])[2]
        decoder_inputs = [self.z, self.decoder_labels]
        self.decoder_outputs = self.decoder_model(decoder_inputs)
        reconstruction_output = Lambda(lambda x: x, name="reconstruction")(self.decoder_outputs)  # 重构数据的输出
        pawine_output = Lambda(lambda x: x, name="pawine")(self.z)
        gears_output = Lambda(lambda x: x, name="gears")(self.decoder_outputs)
        self.cvae_model = Model(inputs=inputs,
                                    outputs=[reconstruction_output, pawine_output, gears_output],
                                    name="cvae")
        self.custom_objects = {'mean_activation': ACTIVATIONS['mean_activation'],
                               'disp_activation': ACTIVATIONS['disp_activation'],
                               'SliceLayer': LAYERS['SliceLayer'],
                               'ColwiseMultLayer': LAYERS['ColWiseMultLayer'],
                               }
        get_custom_objects().update(self.custom_objects)
        print(f"{self.class_name}' network has been successfully constructed!")


    def _calculate_loss(self):

        kl_loss = LOSSES['kl'](self.mu, self.log_var)

        if self.loss_fn == 'nb':
            recon_loss = LOSSES['nb_wo_kl'](self.disp)
        elif self.loss_fn == 'zinb':
            recon_loss = LOSSES['zinb_wo_kl']
        else:
            recon_loss = LOSSES[f'{self.loss_fn}_recon']

        pawine_loss = LOSSES['pawine'](self, self.pawine_lambda, self.pawine_lambda_d)

        loss = LOSSES[self.loss_fn](self.mu, self.log_var, self.alpha, self.eta)

        gears_loss = LOSSES['gears'](
            ctrl_expr=self.ctrl_expr,
            gamma=self.gears_gamma,
            direction_lambda=self.gears_direction_lambda,
            gears_lambda=self.gears_lambda
        )

        return loss, recon_loss, kl_loss, pawine_loss, gears_loss


    def compile_models(self):

        optimizer = keras.optimizers.Adam(lr=self.lr, clipvalue=self.clip_value, epsilon=self.epsilon)

        loss, recon_loss, kl_loss, pawine_loss, gears_loss = self._calculate_loss()

        self.cvae_model.compile(optimizer=optimizer,
                                loss=[loss, pawine_loss, gears_loss],
                                metrics={self.cvae_model.outputs[0].name: loss,
                                         self.cvae_model.outputs[1].name: pawine_loss,
                                         self.cvae_model.outputs[2].name: gears_loss
                                         }
                                )

        print("GOCVAE's network has been successfully compiled!")


    def _prepare_pawine_triplets_for_batch(self, train_adata, batch_indices):

        if not hasattr(self, 'go_similarity_dict') or len(self.go_similarity_dict) == 0:
            self.pawine_triplets = None
            return

        batch_nperts = train_adata.obs['nperts'].iloc[batch_indices].values
        batch_genes = train_adata.obs['perturbation'].iloc[batch_indices].values

        single_gene_mask = (batch_nperts == 1) & (batch_genes != 'control')
        single_indices = np.where(single_gene_mask)[0]

        if len(single_indices) == 0:
            self.pawine_triplets = None
            return

        gene_to_indices = {}
        for i, gene in enumerate(batch_genes):
            if gene != 'control':
                if gene not in gene_to_indices:
                    gene_to_indices[gene] = []
                gene_to_indices[gene].append(i)

        valid_anchors = []
        valid_pos = []
        valid_neg = []
        valid_weights = []

        for anchor_idx in single_indices:
            anchor_gene = batch_genes[anchor_idx]

            if anchor_gene in self.go_similarity_dict:
                similar_genes = self.go_similarity_dict[anchor_gene]
                if not similar_genes:
                    continue

                weights_array = np.array([weight for gene, weight in similar_genes])
                probabilities = weights_array / np.sum(weights_array)

                chosen_idx = np.random.choice(len(similar_genes), p=probabilities)
                pos_gene, edge_weight = similar_genes[chosen_idx]

                if pos_gene in gene_to_indices:
                    pos_idx = gene_to_indices[pos_gene][0]

                    anchor_similar_dict = {g: w for g, w in similar_genes}

                    negative_candidates = []
                    negative_weights = []

                    for gene, indices in gene_to_indices.items():
                        if gene != anchor_gene:
                            if gene in anchor_similar_dict:
                                weight = 1.0 / (anchor_similar_dict[gene] + 0.1)
                            else:
                                weight = getattr(self, 'recommended_unrelated_weight', 10.0)

                            negative_candidates.extend(indices)
                            negative_weights.extend([weight] * len(indices))

                    if negative_candidates:
                        negative_weights = np.array(negative_weights)
                        neg_probabilities = negative_weights / np.sum(negative_weights)
                        neg_idx = np.random.choice(negative_candidates, p=neg_probabilities)
                    else:
                        candidates = [i for i in range(len(batch_genes))
                                      if i != anchor_idx and batch_genes[i] != 'control']
                        if candidates:
                            neg_idx = np.random.choice(candidates)
                        else:
                            continue

                    valid_anchors.append(anchor_idx)
                    valid_pos.append(pos_idx)
                    valid_neg.append(neg_idx)
                    valid_weights.append(edge_weight)

        if valid_anchors:
            self.pawine_triplets = {
                'anchor': valid_anchors,
                'pos': valid_pos,
                'neg': valid_neg,
                'weights': valid_weights
            }
        else:
            self.pawine_triplets = None


    def _sample_go_triplets(self, z, single_gene_mask, batch_genes):

        if self.go_similarity_dict is None:
            return None, None, None

        batch_size = tf.shape(z)[0]
        single_indices = tf.where(single_gene_mask)[:, 0]

        if tf.shape(single_indices)[0] == 0:
            return None, None, None

        pos_indices = tf.random.uniform([tf.shape(single_indices)[0]], 0, batch_size, dtype=tf.int32)
        neg_indices = tf.random.uniform([tf.shape(single_indices)[0]], 0, batch_size, dtype=tf.int32)

        pos_z = tf.gather(z, pos_indices)
        neg_z = tf.gather(z, neg_indices)
        go_weights = tf.ones([tf.shape(single_indices)[0]], dtype=tf.float32)

        return pos_z, neg_z, go_weights

    def _analyze_go_weight_distribution(self):

        if not hasattr(self, 'go_similarity_dict') or len(self.go_similarity_dict) == 0:
            return

        all_weights = []
        for gene, similar_genes in self.go_similarity_dict.items():
            for target_gene, weight in similar_genes:
                all_weights.append(weight)

        all_weights = np.array(all_weights)

        inverse_weights = 1.0 / (all_weights + 0.01)

        self.recommended_unrelated_weight = np.percentile(inverse_weights, 95)


    def to_z_latent(self, adata, batch_key):

        if sparse.issparse(adata.X):
            adata.X = adata.X.A

        encoder_labels, _ = label_encoder(adata, self.condition_encoder, batch_key)
        encoder_labels = encoder_labels.reshape(-1, 1).astype(np.int32)

        latent = self.encoder_model.predict([adata.X, encoder_labels])[2]
        latent = np.nan_to_num(latent)

        latent_adata = anndata.AnnData(X=latent)
        latent_adata.obs = adata.obs.copy(deep=True)

        return latent_adata


    def get_latent(self, adata, batch_key, return_z=True):

        if set(self.gene_names).issubset(set(adata.var_names)):
            adata = adata[:, self.gene_names]
        else:
            raise Exception("set of gene names in train adata are inconsistent with CLDR-CVAE'sgene_names")

        return self.to_z_latent(adata, batch_key)


    def predict(self, adata, condition_key, target_condition, ctrl_key='control'):

        adata = remove_sparsity(adata)

        if ctrl_key in adata.obs.columns:
            ctrl_mask = adata.obs[ctrl_key] == 1
            if ctrl_mask.sum() > 0:
                ctrl_adata = adata[ctrl_mask].copy()
            else:
                ctrl_adata = adata.copy()
        else:
            ctrl_adata = adata.copy()

        n_cells = ctrl_adata.shape[0]
        ctrl_expr = ctrl_adata.X

        if target_condition not in self.condition_encoder:

            if self.use_gears_embedding and self.condition_embedding_layer is not None:
                self.condition_embedding_layer.set_ood_condition(
                    target_condition,
                    self.gene2idx
                )
                pert_idx = 0
            else:
                raise ValueError(
                    f"The OOD condition ‘{target_condition}’ requires GEARS embedded support."
                )
        else:
            pert_idx = self.condition_encoder[target_condition]
            if self.use_gears_embedding and self.condition_embedding_layer is not None:
                self.condition_embedding_layer.ood_gene_indices = None
                self.condition_embedding_layer.ood_gene_mask = None

        encoder_labels = np.full((n_cells, 1), pert_idx, dtype=np.int32)
        decoder_labels = np.full((n_cells, 1), pert_idx, dtype=np.int32)

        pert_expr_placeholder = np.zeros((n_cells, self.gene_size), dtype=np.float32)

        model_output = self.cvae_model.predict(
            [ctrl_expr, encoder_labels, decoder_labels, pert_expr_placeholder]
        )
        x_hat = model_output[0]

        if self.use_gears_embedding and self.condition_embedding_layer is not None:
            self.condition_embedding_layer.ood_gene_indices = None
            self.condition_embedding_layer.ood_gene_mask = None

        adata_pred = anndata.AnnData(X=x_hat)
        adata_pred.obs = ctrl_adata.obs.copy()
        adata_pred.var_names = ctrl_adata.var_names

        adata_pred.obs['predicted_condition'] = target_condition

        return adata_pred


    def restore_model_weights(self, compile=True):

        import h5py

        weight_file = os.path.join(self.model_path, f'{self.model_name}.h5')

        try:
            self.cvae_model.load_weights(weight_file)
        except AttributeError:
            print("Detected Keras version compatibility issues, currently being fixed...")

            with h5py.File(weight_file, 'a') as f:
                if 'keras_version' in f.attrs:
                    keras_version = f.attrs['keras_version']

                    if isinstance(keras_version, bytes):
                        f.attrs['keras_version'] = keras_version.decode('utf8')
                    elif isinstance(keras_version, str):
                        pass
                    else:
                        f.attrs['keras_version'] = str(keras_version)

            self.cvae_model.load_weights(weight_file)

        self.encoder_model = self.cvae_model.get_layer("encoder")
        self.decoder_model = self.cvae_model.get_layer("decoder")

        if compile:
            self.cvae_model.compile(optimizer=self.cvae_model.optimizer,
                                    loss=self.cvae_model.loss,
                                    metrics=self.cvae_model.metrics)


    def restore_model_config(self, compile=True):

        if os.path.exists(os.path.join(self.model_path, f"{self.model_name}.json")):
            json_file = open(os.path.join(self.model_path, f"{self.model_name}.json"), 'rb')
            loaded_model_json = json_file.read()
            self.cvae_model = model_from_json(loaded_model_json)
            self.encoder_model = self.cvae_model.get_layer("encoder")
            self.decoder_model = self.cvae_model.get_layer("decoder")

            if compile:
                self.compile_models()

            print(f"{self.model_name}'s network's config has been successfully restored!")
            return True
        else:
            return False

    def restore_class_config(self, compile_and_consturct=True):

        import json
        if os.path.exists(os.path.join(self.model_path, f"{self.class_name}.json")):
            with open(os.path.join(self.model_path, f"{self.class_name}.json"), 'rb') as f:
                CLDR_config = json.load(f)

            for key, value in CLDR_config.items():
                if key in self.network_kwargs.keys():
                    self.network_kwargs[key] = value
                elif key in self.training_kwargs.keys():
                    self.training_kwargs[key] = value

            for key, value in CLDR_config.items():
                setattr(self, key, value)

            if compile_and_consturct:
                self.construct_network()
                self.compile_models()

            print(f"{self.class_name}'s config has been successfully restored!")
            return True
        else:
            return False

    def save(self, make_dir=True):

        if make_dir:
            os.makedirs(self.model_path, exist_ok=True)

        if os.path.exists(self.model_path):
            self.save_model_weights(make_dir)
            self.save_model_config(make_dir)
            self.save_class_config(make_dir)
            print(f"\n{self.class_name} has been successfully saved in {self.model_path}.")
            return True
        else:
            return False

    def save_model_weights(self, make_dir=True):

        if make_dir:
            os.makedirs(self.model_path, exist_ok=True)

        if os.path.exists(self.model_path):
            self.cvae_model.save_weights(os.path.join(self.model_path, f"{self.model_name}.h5"),
                                         overwrite=True)
            return True
        else:
            return False

    def save_model_config(self, make_dir=True):

        if make_dir:
            os.makedirs(self.model_path, exist_ok=True)

        if os.path.exists(self.model_path):
            model_json = self.cvae_model.to_json()
            with open(os.path.join(self.model_path, f"{self.model_name}.json"), 'w') as file:
                file.write(model_json)
            return True
        else:
            return False

    def save_class_config(self, make_dir=True):

        import json

        if make_dir:
            os.makedirs(self.model_path, exist_ok=True)

        if os.path.exists(self.model_path):
            config = {"gene_size": self.gene_size,
                      "n_topic": self.n_topic,
                      "n_conditions": self.n_conditions,
                      "condition_encoder": self.condition_encoder,
                      "gene_names": self.gene_names}
            all_configs = dict(list(self.network_kwargs.items()) +
                               list(self.training_kwargs.items()) +
                               list(config.items()))
            with open(os.path.join(self.model_path, f"{self.class_name}.json"), 'w') as f:
                json.dump(all_configs, f)

            return True
        else:
            return False

    def _fit(self, adata, condition_key, train_size=0.8,
             n_epochs=300, batch_size=512,
             early_stop_limit=10, lr_reducer=7,
             save=True, retrain=True, verbose=3,
             ctrl_key='control', pert_key='perturbation', nperts_key='nperts'):

        if not hasattr(self, 'go_similarity_dict') or len(self.go_similarity_dict) == 0:
            if 'symbol' in adata.var.columns:
                all_genes = adata.var['symbol'].tolist()
            elif 'gene_name' in adata.var.columns:
                all_genes = adata.var['gene_name'].tolist()
            else:
                all_genes = adata.var_names.tolist()

            filtered_genes = filter_genes_with_go_annotation(all_genes, self.data_path)

            if len(filtered_genes) >= 5:
                self.go_df = get_similarity_network(k=20, data_path=self.data_path,
                                                    data_name=self.data_name,
                                                    pert_list=filtered_genes)
                os.makedirs(self.model_path, exist_ok=True)
                go_path = os.path.join(self.model_path, 'go_graph_shared.csv')
                self.go_df.to_csv(go_path, index=False)
                self.go_graph_path = go_path

                self.go_similarity_dict = {}
                for _, row in self.go_df.iterrows():
                    source = row['source']
                    if source not in self.go_similarity_dict:
                        self.go_similarity_dict[source] = []
                    self.go_similarity_dict[source].append((row['target'], row['importance']))

                print(f"GO网络初始化完成：{len(self.go_similarity_dict)}个节点")
                self._analyze_go_weight_distribution()
            else:
                self.go_similarity_dict = {}

        train_adata, valid_adata = train_test_split(adata, train_size)

        if self.gene_names is None:
            self.gene_names = train_adata.var_names.tolist()
        else:
            if set(self.gene_names).issubset(set(train_adata.var_names)):
                train_adata = train_adata[:, self.gene_names]
            else:
                raise Exception("set of gene names in train adata are inconsistent with class' gene_names")
            if set(self.gene_names).issubset(set(valid_adata.var_names)):
                valid_adata = valid_adata[:, self.gene_names]
            else:
                raise Exception("set of gene names in valid adata are inconsistent with class' gene_names")

        _, self.condition_encoder = label_encoder(train_adata, le=self.condition_encoder,
                                                  condition_key=condition_key)
        _, self.condition_encoder = label_encoder(valid_adata, le=self.condition_encoder,
                                                  condition_key=condition_key)

        train_dataset = PerturbationDataset(
            train_adata,
            condition_key=condition_key,
            ctrl_key=ctrl_key,
            pert_key=pert_key,
            nperts_key=nperts_key
        )

        use_smart_batches = False
        if self.go_similarity_dict:
            print("Building intelligent training batches...")
            try:
                train_dataset.build_smart_batches_for_pawine(
                    self.go_similarity_dict,
                    batch_size=batch_size,
                    single_gene_ratio=0.6,
                    multi_gene_ratio=0.3,
                    ctrl_ratio=0.1,
                    go_similarity_threshold=self.go_similarity_threshold
                )
                use_smart_batches = True
            except Exception as e:
                use_smart_batches = False

        if not retrain and os.path.exists(os.path.join(self.model_path, f"{self.model_name}.h5")):
            self.restore_model_weights()
            return

        callbacks = [History()]
        if verbose > 2:
            callbacks.append(
                LambdaCallback(on_epoch_end=lambda epoch, logs:
                print_progress(epoch, logs, n_epochs)))
            fit_verbose = 0
        else:
            fit_verbose = verbose

        if early_stop_limit > 0:
            callbacks.append(EarlyStopping(patience=early_stop_limit, monitor='val_loss'))
        if lr_reducer > 0:
            callbacks.append(ReduceLROnPlateau(monitor='val_loss', patience=lr_reducer))

        train_generator = train_dataset.create_train_generator(
            self.condition_encoder,
            batch_size=batch_size,
            use_smart_batches=use_smart_batches
        )

        x_valid, y_valid = train_dataset.get_validation_data(valid_adata, self.condition_encoder)

        if use_smart_batches:
            steps_per_epoch = len(train_dataset.smart_batches)
        else:
            steps_per_epoch = train_dataset.n_conditions

        self.cvae_model.fit_generator(
            generator=train_generator(),
            steps_per_epoch=steps_per_epoch,
            validation_data=(x_valid, y_valid),
            epochs=n_epochs,
            callbacks=callbacks,
            verbose=fit_verbose,
            max_queue_size=1,
            workers=1,
            use_multiprocessing=False
        )

        if save:
            self.save(make_dir=True)


    def _train_on_batch(self, adata,
                        condition_key, train_size=0.8,
                        n_epochs=300, batch_size=512,
                        early_stop_limit=10,
                        save=True, retrain=True):
        train_adata, valid_adata = train_test_split(adata, train_size)

        if self.gene_names is None:
            self.gene_names = train_adata.var_names.tolist()
        else:
            if set(self.gene_names).issubset(set(train_adata.var_names)):
                train_adata = train_adata[:, self.gene_names]
            else:
                raise Exception("set of gene names in train adata are inconsistent with class' gene_names")
            if set(self.gene_names).issubset(set(valid_adata.var_names)):
                valid_adata = valid_adata[:, self.gene_names]
            else:
                raise Exception("set of gene names in valid adata are inconsistent with class' gene_names")

        train_conditions_encoded, _ = label_encoder(train_adata, le=self.condition_encoder,
                                                                         condition_key=condition_key)
        valid_conditions_encoded, _ = label_encoder(valid_adata, le=self.condition_encoder,
                                                                         condition_key=condition_key)

        if not retrain and os.path.exists(os.path.join(self.model_path, f"{self.model_name}.h5")):
            self.restore_model_weights()
            return

        train_conditions_indices = train_conditions_encoded.reshape(-1, 1).astype(np.int32)
        valid_conditions_indices = valid_conditions_encoded.reshape(-1, 1).astype(np.int32)

        if sparse.issparse(train_adata.X):
            is_sparse = True
        else:
            is_sparse = False

        train_expr = train_adata.X
        valid_expr = valid_adata.X.A if is_sparse else valid_adata.X

        x_valid = [valid_expr, valid_conditions_indices, valid_conditions_indices]

        if self.loss_fn in ['nb', 'zinb']:
            x_valid.append(valid_adata.obs[self.size_factor_key].values)
            pawine_placeholder = np.zeros((batch_size, self.n_topic))
            y_valid = [valid_adata.raw.X.A if sparse.issparse(valid_adata.raw.X) else valid_adata.raw.X,
                       valid_conditions_encoded, pawine_placeholder]
        else:
            pawine_placeholder = np.zeros((batch_size, self.n_topic))
            y_valid = [valid_expr, valid_conditions_encoded, pawine_placeholder]

        es_patience, best_val_loss = 0, 1e10
        for i in range(n_epochs):
            train_loss = train_recon_loss = train_contrastive_loss = train_second_contrastive_loss = 0.0

            steps_per_epoch = max(1, train_adata.shape[0] // batch_size)

            for j in range(steps_per_epoch):
                batch_indices = np.random.choice(train_adata.shape[0], batch_size)
                batch_expr = train_expr[batch_indices, :].A if is_sparse else train_expr[batch_indices, :]

                x_train = [batch_expr, train_conditions_indices[batch_indices], train_conditions_indices[batch_indices]]

                if self.loss_fn in ['nb', 'zinb']:
                    x_train.append(train_adata.obs[self.size_factor_key].values[batch_indices])
                    y_train = [train_adata.raw.X[batch_indices].A if sparse.issparse(
                        train_adata.raw.X[batch_indices]) else train_adata.raw.X[batch_indices], pawine_placeholder]
                else:
                    y_train = [batch_expr, pawine_placeholder]

                batch_loss, batch_recon_loss, batch_contrastive_loss, batch_second_contrastive_loss = self.cvae_model.train_on_batch(x_train, y_train)

                train_loss += batch_loss / batch_size
                train_recon_loss += batch_recon_loss / batch_size
                train_contrastive_loss += batch_contrastive_loss / batch_size
                train_second_contrastive_loss += batch_second_contrastive_loss / batch_size

            valid_loss, valid_recon_loss, valid_contrastive_loss, valid_second_contrastive_loss = self.cvae_model.evaluate(
                x_valid, y_valid, verbose=0)

            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                es_patience = 0
            else:
                es_patience += 1
                if es_patience == early_stop_limit:
                    print("Training stopped with Early Stopping")
                    break

            logs = {"loss": train_loss, "recon_loss": train_recon_loss,
                    "contrastive_loss": train_contrastive_loss,
                    "second_contrastive_loss": train_second_contrastive_loss,
                    "val_loss": valid_loss, "val_recon_loss": valid_recon_loss,
                    "val_contrastive_loss": valid_contrastive_loss,
                    "val_second_contrastive_loss": valid_second_contrastive_loss}
            print_progress(i, logs, n_epochs)

        if save:
            self.save(make_dir=True)


    def train(self, adata,
              condition_key, train_size=0.8,
              n_epochs=200, batch_size=128,
              early_stop_limit=10, lr_reducer=8,
              save=True, retrain=True, verbose=3):

        if self.device == 'gpu':
            return self._fit(adata, condition_key, train_size, n_epochs, batch_size, early_stop_limit,
                             lr_reducer, save, retrain, verbose,
                             ctrl_key=self.ctrl_key, pert_key=self.pert_key, nperts_key=self.nperts_key)
        else:
            return self._train_on_batch(adata, condition_key, train_size, n_epochs, batch_size,
                                        early_stop_limit, lr_reducer, save, retrain, verbose)