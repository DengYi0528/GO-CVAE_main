import tensorflow as tf
from tensorflow.keras import backend as K
from keras.layers import Layer
from keras import backend as K
import numpy as np
from GOCVAE.utils import parse_any_pert



class SliceLayer(Layer):
    def __init__(self, index=0, **kwargs):
        self.index = index
        super().__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('Input should be a list')

        super().build(input_shape)

    def call(self, x, **kwargs):
        assert isinstance(x, list), 'SliceLayer input is not a list'
        return x[self.index]

    def compute_output_shape(self, input_shape):
        return input_shape[self.index]


class ColwiseMultLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('Input should be a list')

        super().build(input_shape)

    def call(self, x, **kwargs):
        assert isinstance(x, list), 'SliceLayer input is not a list'
        return x[0] * K.reshape(x[1], (-1, 1))

    def compute_output_shape(self, input_shape):
        return input_shape[0]


LAYERS = {
    "SliceLayer": SliceLayer,
    "ColWiseMultLayer": ColwiseMultLayer,
}


class GraphConvLayer(Layer):

    def __init__(self, output_dim, adj_matrix, **kwargs):

        self.output_dim = output_dim

        from scipy.sparse import coo_matrix
        adj_coo = coo_matrix(adj_matrix)

        self.adj_indices = np.column_stack((adj_coo.row, adj_coo.col)).astype(np.int64)
        self.adj_values = adj_coo.data.astype(np.float32)
        self.adj_shape = adj_coo.shape

        super(GraphConvLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        super(GraphConvLayer, self).build(input_shape)

    def call(self, x):

        adj_sparse = tf.SparseTensor(
            indices=self.adj_indices,
            values=self.adj_values,
            dense_shape=self.adj_shape
        )

        aggregated = tf.sparse.sparse_dense_matmul(adj_sparse, x)

        output = K.dot(aggregated, self.kernel)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class ConditionEmbeddingLayer(Layer):

    def __init__(self, conditions, condition_to_genes, all_genes, gene2idx,
                 go_adj_matrix, embedding_dim=64, num_gnn_layers=1,
                 max_genes_per_condition=5, **kwargs):

        self.conditions = conditions
        self.all_genes = all_genes
        self.gene2idx = gene2idx
        self.num_all_genes = len(all_genes)
        self.embedding_dim = embedding_dim
        self.num_gnn_layers = num_gnn_layers
        self.max_genes_per_condition = max_genes_per_condition
        self.ood_gene_indices = None
        self.ood_gene_mask = None

        condition_gene_matrix = []
        condition_gene_mask = []

        for cond in conditions:
            genes = condition_to_genes[cond]

            gene_indices = []
            for g in genes:
                if g in gene2idx:
                    gene_indices.append(gene2idx[g])

            padded_indices = gene_indices + [-1] * (max_genes_per_condition - len(gene_indices))
            padded_indices = padded_indices[:max_genes_per_condition]

            mask = [1.0] * len(gene_indices) + [0.0] * (max_genes_per_condition - len(gene_indices))
            mask = mask[:max_genes_per_condition]

            condition_gene_matrix.append(padded_indices)
            condition_gene_mask.append(mask)

        self.condition_gene_matrix = np.array(condition_gene_matrix, dtype=np.int32)
        self.condition_gene_mask = np.array(condition_gene_mask, dtype=np.float32)

        self.go_adj_matrix = go_adj_matrix

        super(ConditionEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.pert_embedding = self.add_weight(
            name='pert_embedding',
            shape=(self.num_all_genes, self.embedding_dim),  # [5045, 64]
            initializer='glorot_uniform',
            trainable=True
        )

        self.gnn_kernels = []
        for i in range(self.num_gnn_layers):
            kernel = self.add_weight(
                name=f'gnn_kernel_{i}',
                shape=(self.embedding_dim, self.embedding_dim),
                initializer='glorot_uniform',
                trainable=True
            )
            self.gnn_kernels.append(kernel)

        self.mlp_w1 = self.add_weight(
            name='mlp_w1',
            shape=(self.embedding_dim, self.embedding_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.mlp_b1 = self.add_weight(
            name='mlp_b1',
            shape=(self.embedding_dim,),
            initializer='zeros',
            trainable=True
        )

        self.mlp_w2 = self.add_weight(
            name='mlp_w2',
            shape=(self.embedding_dim, self.embedding_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.mlp_b2 = self.add_weight(
            name='mlp_b2',
            shape=(self.embedding_dim,),
            initializer='zeros',
            trainable=True
        )

        from scipy.sparse import coo_matrix
        adj_coo = coo_matrix(self.go_adj_matrix)
        self.adj_indices = adj_coo.row.astype(np.int64), adj_coo.col.astype(np.int64)
        self.adj_values = adj_coo.data.astype(np.float32)
        self.adj_shape = adj_coo.shape

        super(ConditionEmbeddingLayer, self).build(input_shape)

    def set_ood_condition(self, ood_condition_name, gene2idx):

        from GOCVAE.utils import parse_any_pert
        import numpy as np

        genes = parse_any_pert(ood_condition_name)
        if genes is None:
            genes = []

        gene_indices = []
        for g in genes:
            if g in gene2idx:
                gene_indices.append(gene2idx[g])

        padded_indices = gene_indices + [-1] * (self.max_genes_per_condition - len(gene_indices))
        padded_indices = padded_indices[:self.max_genes_per_condition]

        mask = [1.0] * len(gene_indices) + [0.0] * (self.max_genes_per_condition - len(gene_indices))
        mask = mask[:self.max_genes_per_condition]

        self.ood_gene_indices = np.array(padded_indices, dtype=np.int32)
        self.ood_gene_mask = np.array(mask, dtype=np.float32)


    def call(self, inputs):

        import tensorflow as tf
        from keras import backend as K

        condition_indices = K.cast(K.flatten(inputs), 'int32')

        all_gene_emb = self.pert_embedding

        enhanced_emb = all_gene_emb
        for i in range(self.num_gnn_layers):
            adj_sparse = tf.SparseTensor(
                indices=tf.transpose(tf.stack([
                    tf.constant(self.adj_indices[0], dtype=tf.int64),
                    tf.constant(self.adj_indices[1], dtype=tf.int64)
                ])),
                values=tf.constant(self.adj_values, dtype=tf.float32),
                dense_shape=self.adj_shape
            )

            aggregated = tf.sparse.sparse_dense_matmul(adj_sparse, enhanced_emb)

            enhanced_emb = K.dot(aggregated, self.gnn_kernels[i])

            if i < self.num_gnn_layers - 1:
                enhanced_emb = K.relu(enhanced_emb)

        if hasattr(self, 'ood_gene_indices') and self.ood_gene_indices is not None:

            batch_size = tf.shape(condition_indices)[0]

            batch_gene_indices = tf.tile(
                tf.expand_dims(tf.constant(self.ood_gene_indices, dtype=tf.int32), 0),
                [batch_size, 1]
            )  # [B, max_genes_per_condition]

            batch_gene_mask = tf.tile(
                tf.expand_dims(tf.constant(self.ood_gene_mask, dtype=tf.float32), 0),
                [batch_size, 1]
            )  # [B, max_genes_per_condition]
        else:
            batch_gene_indices = tf.gather(
                tf.constant(self.condition_gene_matrix, dtype=tf.int32),
                condition_indices
            )  # [B, max_genes_per_condition]

            batch_gene_mask = tf.gather(
                tf.constant(self.condition_gene_mask, dtype=tf.float32),
                condition_indices
            )  # [B, max_genes_per_condition]

        batch_gene_indices_safe = tf.maximum(batch_gene_indices, 0)

        batch_gene_embs = tf.gather(enhanced_emb, batch_gene_indices_safe)

        mask_expanded = tf.expand_dims(batch_gene_mask, axis=-1)  # [B, max_genes, 1]
        batch_gene_embs_masked = batch_gene_embs * mask_expanded  # [B, max_genes, 64]

        summed_embs = tf.reduce_sum(batch_gene_embs_masked, axis=1)  # [B, 64]

        num_genes_per_sample = tf.reduce_sum(batch_gene_mask, axis=1)  # [B]

        num_genes_per_sample = tf.reduce_sum(batch_gene_mask, axis=1)  # [B]

        def apply_mlp(x):
            h = K.dot(x, self.mlp_w1) + self.mlp_b1
            h = K.relu(h)
            h = K.dot(h, self.mlp_w2) + self.mlp_b2
            h = K.relu(h)
            return h

        def process_single_sample(args):

            summed_emb, n_genes = args

            def return_zero():
                return tf.zeros((self.embedding_dim,), dtype=tf.float32)

            def return_single():
                return summed_emb

            def return_multi():
                return apply_mlp(tf.expand_dims(summed_emb, 0))[0]

            result = tf.case(
                [
                    (tf.equal(n_genes, 0.0), return_zero),
                    (tf.equal(n_genes, 1.0), return_single),
                ],
                default=return_multi
            )

            return result

        output = tf.map_fn(
            process_single_sample,
            (summed_embs, num_genes_per_sample),
            dtype=tf.float32,
            parallel_iterations=10
        )  # [B, 64]

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.embedding_dim)