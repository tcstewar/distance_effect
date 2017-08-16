import nengo
import nengo.spa as spa
import numpy as np
import pytry

class DistanceEffectTrial(pytry.NengoTrial):
    def params(self):
        self.param('dimensions', D=16)
        self.param('number of neurons', n_neurons=100)
        self.param('inter-stimulus interval', isi=0.5)
        self.param('probe synapse', probe_synapse=0.1)
        self.param('representation mode (concat|convolve)', rep_mode='concat')
        self.param('training noise', noise=0.0)
        self.param('training copies', copies=1)
        self.param('equal number of trials across distances', balance_trials_by_distance=False)

    def model(self, p):

        digits = ['ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE']

        vocab = spa.Vocabulary(p.D)

        model = nengo.Network()
        with model:


            if p.rep_mode == 'concat':
                dims = p.D * 2
            elif p.rep_mode == 'convolve':
                dims = p.D
            else:
                raise ValueError('rep_mode must be "concat" or "convolve"')

            ens = nengo.Ensemble(n_neurons=p.n_neurons, dimensions=dims)

            answer = nengo.Ensemble(n_neurons=100, dimensions=1,
                                    neuron_type=nengo.Direct())


            inputs = []
            outputs = []
            distance = []
            self.inputs = inputs
            self.outputs = outputs
            self.distance = distance

            items = []
            if not p.balance_trials_by_distance:
                for i in range(len(digits)):
                    for j in range(len(digits)):
                        if i != j:
                            items.append((i,j))
            else:
                for dist in range(1,9):
                    i = 0
                    for item in range(8):
                        items.append((i, i+dist))
                        items.append((i+dist, i))
                        i += 1
                        if i+dist >= len(digits):
                            i = 0

            for i,j in items:
                        if p.rep_mode == 'concat':
                            n1 = vocab.parse(digits[i]).v
                            n2 = vocab.parse(digits[j]).v
                            v = np.hstack([n1, n2])
                        elif p.rep_mode == 'convolve':
                            txt = 'OBJ1*%s + OBJ2*%s' % (digits[i], digits[j])
                            v = vocab.parse(txt).v
                        inputs.append(v)
                        if i < j:
                            outputs.append([-1])
                        else:
                            outputs.append([1])
                        distance.append(abs(i-j))

            training_inputs = np.tile(self.inputs, (p.copies, 1))
            training_outputs = np.tile(self.outputs, (p.copies, 1))
            if p.noise > 0:
                training_inputs = np.random.normal(training_inputs, p.noise)
                training_outputs = np.random.normal(training_outputs, p.noise)

            nengo.Connection(ens, answer, function=training_outputs, eval_points=training_inputs)

            def stimulus(t):
                index = int(t / p.isi) % len(inputs)
                return self.inputs[index]
            stim = nengo.Node(stimulus)
            nengo.Connection(stim, ens)

            self.p_answer = nengo.Probe(answer, synapse=p.probe_synapse)

        return model

    def evaluate(self, p, sim, plt):
        order = np.arange(len(self.inputs))
        np.random.shuffle(order)
        self.inputs = np.array(self.inputs)[order]
        self.outputs = np.array(self.outputs)[order]
        self.distance = np.array(self.distance)[order]

        sim.run(p.isi * len(self.inputs))

        data = sim.data[self.p_answer]

        steps = int(p.isi / p.dt)
        final_data = data[steps-1::steps]

        error = {
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
            8: [],
            }

        for i in range(len(final_data)):
            answer = final_data[i]
            correct = self.outputs[i]
            distance = self.distance[i]
            error[distance].append(abs(float(answer-correct)))

        average = {k: np.mean(v) for k,v in error.items()}

        if plt:
            plt.plot(sim.trange(), data)

        return dict(error=error,
                    average=average)


