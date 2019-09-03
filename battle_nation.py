from tensorflow import keras as keras
import numpy as np

RANDOMNESS_SIZE = 2
BATCH_SIZE = 20


def strategy_definition(tensor, strategy_name):
    normalize_input = keras.layers.Multiply(name='{}_probability_normalisation'.format(strategy_name))([tensor, 1 / keras.backend.sum(tensor, axis=1)])
    distribute_soliders = keras.layers.Lambda(lambda t: 100 * t, name='{}_soldier_distribution'.format(strategy_name))(normalize_input)
    return distribute_soliders
    
def build_strategy(strategy_input, strategy_name):
    dense_layer = keras.layers.Dense(10, activation='sigmoid')(strategy_input)
    strategy = strategy_definition(dense_layer, strategy_name)
    return strategy
    
   
def value_points(input_batch):
    tf_constant = keras.backend.constant(CASTLE_VALUES.reshape(1, -1))
    batch_size = keras.backend.shape(input_batch)[0]
    tiled_constant = keras.backend.tile(tf_constant, (batch_size, 1))
    return keras.layers.dot([input_batch, tiled_constant], axes=1)
    
def game_layer(input_strategy_1, input_strategy_2):
    combat = keras.layers.Subtract(name='point_difference')([input_strategy_1, input_strategy_2])
    
    rounds_won = keras.layers.Lambda(lambda t: 2 * keras.backend.sigmoid(5 * t) - 1)(combat)
#     points_scored = keras.layers.Lambda(lambda t: keras.backend.sum(t))(rounds_won)
    points_scored = value_points(rounds_won)
    return keras.layers.Lambda(lambda t: keras.backend.sigmoid(5 * t))(points_scored)
    
strategy_1_seed = keras.layers.Input((RANDOMNESS_SIZE,), name='strategy_1_input')
strategy_2_seed = keras.layers.Input((RANDOMNESS_SIZE,), name='strategy_2_input')

strategy_1 = keras.Model(
    inputs=strategy_1_seed, outputs=[build_strategy(strategy_1_seed, 'first_strat')])
strategy_2 = keras.Model(
    inputs=strategy_2_seed, outputs=[build_strategy(strategy_2_seed, 'first_strat')])
    
strategy_1_seed = keras.layers.Input((RANDOMNESS_SIZE,), name='strategy_1_input')
strategy_2_seed = keras.layers.Input((RANDOMNESS_SIZE,), name='strategy_2_input')

trainable_game_result = game_layer(
    strategy_1(strategy_1_seed), 
    strategy_2(strategy_2_seed)
)

strategy_1.trainable = True
strategy_2.trainable = False

trainable_game_1 = keras.Model(inputs=[strategy_1_seed, strategy_2_seed], outputs=trainable_game_result)
trainable_game_1.compile('adam', loss='binary_crossentropy')

strategy_1.trainable = False
strategy_2.trainable = True

trainable_game_2 = keras.Model(inputs=[strategy_1_seed, strategy_2_seed], outputs=trainable_game_result)
trainable_game_2.compile('adam', loss='binary_crossentropy')

for epoch in range(10000):
    if epoch % 1000 == 0:
        print(epoch)
    
    trainable_game_1.train_on_batch(
        [np.random.normal(0, 1, [BATCH_SIZE, RANDOMNESS_SIZE]), 
         np.random.normal(0, 1, [BATCH_SIZE, RANDOMNESS_SIZE])], 
        np.ones((BATCH_SIZE,))
    )
    
    trainable_game_2.train_on_batch(
        [np.random.normal(0, 1, [BATCH_SIZE, RANDOMNESS_SIZE]), 
         np.random.normal(0, 1, [BATCH_SIZE, RANDOMNESS_SIZE])], 
        np.zeros((BATCH_SIZE,))
    )
