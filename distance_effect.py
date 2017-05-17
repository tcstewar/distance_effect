import nengo
import nengo.spa as spa
import numpy as np


digits = ['ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE']

D = 16
vocab = spa.Vocabulary(D)

model = nengo.Network()
with model:
    model.config[nengo.Ensemble].neuron_type=nengo.Direct()
    num1 = spa.State(D, vocab=vocab)
    num2 = spa.State(D, vocab=vocab)
    
    model.config[nengo.Ensemble].neuron_type=nengo.LIF()
    ens = nengo.Ensemble(n_neurons=100, dimensions=D*2)
    
    model.config[nengo.Ensemble].neuron_type=nengo.Direct()
    answer = nengo.Ensemble(n_neurons=100, dimensions=1)
    
    nengo.Connection(num1.output, ens[:D]) # connect to the first D dimensions
    nengo.Connection(num2.output, ens[D:]) # connect to the second D dimensions
    
    inputs = []
    outputs = []
    
    for i in range(len(digits)):
        for j in range(len(digits)):
            if i != j:
                n1 = vocab.parse(digits[i]).v
                n2 = vocab.parse(digits[j]).v
                v = np.hstack([n1, n2])
                inputs.append(v)
                if i < j:
                    outputs.append([-1])
                else:
                    outputs.append([1])
                
    nengo.Connection(ens, answer, function=outputs, eval_points=inputs)
    