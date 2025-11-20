"""
This module defines the HybridNCF architecture that combines:
- Collaborative Filtering: User-item interaction patterns from ratings
- Content-Based Filtering: Pre-trained BERT embeddings from review text
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import register_keras_serializable
import numpy as np

@register_keras_serializable(package="MyModels") 
class HybridNCF(Model):
    """Hybrid Neural Collaborative Filtering

    Combines collaborative filtering (user-item interactions) with content-based
    filtering (BERT embeddings from review text) for improved recommendation accuracy.

    Architecture:
    - User_id CF embeddings (learned from ratings)
    - Item_id CF embeddings (learned from ratings)
    - Item text content embeddings (using pre-trained BERT)
    - CF interaction term (element-wise multiplication)
    - Deep fusion network with batch normalization

    Args:
        n_users: Number of unique users
        n_items: Number of unique items
        pretrained_item_embeddings: Pre-trained BERT embeddings for items (optional)
        embedding_dim: Dimension of CF embeddings (default: 50)
        mlp_layers: List of hidden layer sizes (default: [128, 64, 32])
        dropout_rate: Dropout rate for regularization (default: 0.2)
        freeze_pretrained: Whether to freeze BERT embeddings (default: True)
        l2_reg: L2 regularization strength (default: 1e-6)
    """

    def __init__(self, n_users, n_items, pretrained_item_embeddings=None, embedding_dim=50,
                 mlp_layers=[128, 64, 32], dropout_rate=0.2, freeze_pretrained=True, l2_reg=1e-6, **kwargs):
        super(HybridNCF, self).__init__(**kwargs)

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.mlp_layers = mlp_layers
        self.dropout_rate = dropout_rate
        self.freeze_pretrained = freeze_pretrained
        self.l2_reg = l2_reg

        self.user_bias = layers.Embedding(n_users, 1)
        self.item_bias = layers.Embedding(n_items, 1)

        # Collaborative filtering embeddings (learned from rating behavior)
        self.user_cf_embedding = layers.Embedding(
            n_users, embedding_dim,
            embeddings_initializer='he_normal',
            embeddings_regularizer=l2(l2_reg),
            name='user_cf_embedding'
        )
        self.item_cf_embedding = layers.Embedding(
            n_items, embedding_dim,
            embeddings_initializer='he_normal',
            embeddings_regularizer=l2(l2_reg),
            name='item_cf_embedding'
        )

        # Content-based embeddings (pre-trained BERT from review text)
        if pretrained_item_embeddings is not None:
            pretrained_dim = pretrained_item_embeddings.shape[1]
            self.item_content_embedding = layers.Embedding(
                n_items, pretrained_dim,
                embeddings_initializer=keras.initializers.Constant(pretrained_item_embeddings),
                trainable=not freeze_pretrained,
                name='item_content_embedding'
            )
            self.has_pretrained = True
            self.pretrained_dim = pretrained_dim
            # Project BERT embeddings to match CF embedding dimension
            self.content_projection = layers.Dense(
                embedding_dim,
                activation='relu',
                kernel_initializer='he_normal',
                name='content_projection'
            )
        else:
            self.has_pretrained = False
            self.pretrained_dim = None

        # Fusion network: combines CF and content signals
        self.fusion_layers = []
        for i, units in enumerate(mlp_layers):
            self.fusion_layers.append(
                layers.Dense(
                    units,
                    activation='relu',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_reg),
                    name=f'fusion_dense_{i}'
                )
            )
            self.fusion_layers.append(
                layers.BatchNormalization(name=f'fusion_bn_{i}')
            )
            self.fusion_layers.append(
                layers.Dropout(dropout_rate, name=f'fusion_dropout_{i}')
            )

        self.output_layer = layers.Dense(1, name='output')

    def call(self, inputs, training=False):
        """Forward pass through the network

        Args:
            inputs: Dictionary with keys 'user' and 'item' containing IDs
            training: Boolean indicating training mode (for dropout/batchnorm)

        Returns:
            Predicted ratings (batch_size, 1)
        """
        user_input = inputs['user']
        item_input = inputs['item']

        # Get CF embeddings
        user_cf = self.user_cf_embedding(user_input)
        item_cf = self.item_cf_embedding(item_input)

        # Compute CF interaction (element-wise multiplication)
        cf_interaction = user_cf * item_cf

        # Combine CF and content signals
        if self.has_pretrained:
            item_content = self.item_content_embedding(item_input)
            item_content_proj = self.content_projection(item_content)
            combined = tf.concat([user_cf, item_cf, item_content_proj, cf_interaction], axis=1)
        else:
            combined = tf.concat([user_cf, item_cf, cf_interaction], axis=1)

        # Pass through fusion network
        x = combined
        for layer in self.fusion_layers:
            if isinstance(layer, layers.Dropout):
                x = layer(x, training=training)
            elif isinstance(layer, layers.BatchNormalization):
                x = layer(x, training=training)
            else:
                x = layer(x)

        # Output prediction
        prediction = self.output_layer(x) + self.user_bias(user_input) + self.item_bias(item_input)
        return prediction

    def get_config(self):
        """Get model configuration for serialization"""
        config = {
            'n_users': self.n_users,
            'n_items': self.n_items,
            'embedding_dim': self.embedding_dim,
            'mlp_layers': self.mlp_layers,
            'dropout_rate': self.dropout_rate,
            'freeze_pretrained': self.freeze_pretrained,
            'l2_reg': self.l2_reg
        }
        
        # Remembers whether Bert embeddings present
        if self.has_pretrained:
            config['pretrained_dim'] = self.pretrained_dim
        
        return config
    
    @classmethod  
    def from_config(cls, config):
        """Create model from config during loading"""
        
        pretrained_dim = config.pop('pretrained_dim', None)
        if pretrained_dim is not None:
            # Create dummy embeddings with the right shape
            # The actual values will be loaded from saved weights
            n_items = config['n_items']
            dummy_embeddings = np.zeros((n_items, pretrained_dim), dtype=np.float32)
            config['pretrained_item_embeddings'] = dummy_embeddings
        
        return cls(**config)

def create_model(n_users, n_items, pretrained_embeddings=None, config=None):
    """Factory function to create Hybrid NCF model

    Args:
        n_users: Number of unique users
        n_items: Number of unique items
        pretrained_embeddings: Pre-trained BERT embeddings for items (optional)
        config: Dictionary with hyperparameters:
            - embedding_dim: CF embedding dimension (default: 50)
            - mlp_layers: List of layer sizes (default: [128, 64, 32])
            - dropout_rate: Dropout rate (default: 0.2)
            - freeze_pretrained: Freeze BERT embeddings (default: True)
            - l2_reg: L2 regularization (default: 1e-6)

    Returns:
        Compiled HybridNCF model instance
    """
    if config is None:
        config = {}

    model = HybridNCF(
        n_users=n_users,
        n_items=n_items,
        pretrained_item_embeddings=pretrained_embeddings,
        embedding_dim=config.get('embedding_dim', 50),
        mlp_layers=config.get('mlp_layers', [128, 64, 32]),
        dropout_rate=config.get('dropout_rate', 0.2),
        freeze_pretrained=config.get('freeze_pretrained', True),
        l2_reg=config.get('l2_reg', 1e-6)
    )

    return model