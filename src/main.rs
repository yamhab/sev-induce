use std::{cmp::Ordering, range::Range};

const POPULATION_COUNT: usize = 1000;
const TOPOLOGY: [usize; 3] = [2, 2, 1];
const ELITE_COUNT: usize = 100;

const MUTATION_RATE: f64 = 0.5;
const MUTATION_STRENGTH: f64 = 0.5;
const MUTATION_RANGE: f64 = 3.0;

const TRIALS: usize = 1000;
const XOR_INPUTS_SET: [[f64; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];

struct Population {
    networks: Vec<Network>,
}

impl Population {
    fn new(topology: &[usize], population_count: usize) -> Self {
        let networks = vec![Network::new(topology); population_count];
        Self { networks }
    }

    fn mutate(&mut self, elite: usize, rate: f64, strength: f64, range: f64) {
        for network in self.networks[elite..].iter_mut() {
            network.mutate(rate, strength, range);
        }
    }

    fn evaluate(&mut self, inputs_set: &[[f64; 2]]) {
        for network in self.networks.iter_mut() {
            network.fitness = 0.0;
            for inputs in inputs_set.iter() {
                let outputs = network.infer(inputs);
                network.fitness +=
                    fitness_function(inputs, &network.buffer[outputs.start..outputs.end]);
            }
        }
    }

    fn select(&mut self, elite: usize) {
        self.networks
            .sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        for i in elite..self.networks.len() {
            self.networks[i] = self.networks[i % elite].clone();
        }
    }
}

#[derive(Clone)]
struct Network {
    buffer: Vec<f64>,
    layers: Vec<Layer>,
    fitness: f64,
}

impl Network {
    fn new(topology: &[usize]) -> Network {
        let mut size: usize = 0;
        for i in 0..topology.len() - 1 {
            size += topology[i + 1] * (topology[i] + 2);
        }
        let buffer = vec![0.0; size];

        let mut layers: Vec<Layer> = Vec::with_capacity(topology.len() - 1);
        let mut offset: usize = 0;
        let mut inputs = Range::from(0..0);
        for i in 0..topology.len() - 1 {
            let input_count = topology[i];
            let output_count = topology[i + 1];

            let weights = Range::from(offset..(offset + input_count * output_count));
            offset += input_count * output_count;
            let biases = Range::from(offset..(offset + output_count));
            offset += output_count;
            let outputs = Range::from(offset..(offset + output_count));
            offset += output_count;

            layers.push(Layer::new(inputs, weights, biases, outputs));
            inputs = outputs;
        }

        Network {
            buffer,
            layers,
            fitness: 0.0,
        }
    }

    fn mutate(&mut self, rate: f64, strength: f64, range: f64) {
        for element in self.buffer.iter_mut() {
            let chance = fastrand::f64();
            if chance < rate {
                let tweak = (fastrand::f64() * 2.0 - 1.0) * strength;
                *element = f64::clamp(*element + tweak, -range, range);
            }
        }
    }

    fn infer(&mut self, inputs: &[f64]) -> Range<usize> {
        self.layers[0].forward(&mut self.buffer, Some(inputs));
        for i in 1..self.layers.len() {
            self.layers[i].forward(&mut self.buffer, None);
        }
        self.layers[self.layers.len() - 1].outputs
    }
}

impl PartialEq for Network {
    fn eq(&self, other: &Self) -> bool {
        self.fitness == other.fitness
    }
}

impl PartialOrd for Network {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.fitness > other.fitness {
            Some(Ordering::Greater)
        } else if self.fitness == other.fitness {
            Some(Ordering::Equal)
        } else if self.fitness < other.fitness {
            Some(Ordering::Less)
        } else {
            None
        }
    }
}

#[derive(Clone)]
struct Layer {
    inputs: Range<usize>,
    weights: Range<usize>,
    biases: Range<usize>,
    outputs: Range<usize>,
}

impl Layer {
    fn new(
        inputs: Range<usize>,
        weights: Range<usize>,
        biases: Range<usize>,
        outputs: Range<usize>,
    ) -> Self {
        Self {
            inputs,
            weights,
            biases,
            outputs,
        }
    }

    fn forward(&mut self, buffer: &mut [f64], inputs: Option<&[f64]>) {
        match inputs {
            Some(inputs) => {
                for (i, output) in self.outputs.into_iter().enumerate() {
                    buffer[output] = buffer[self.biases.start + i];
                    for (j, input) in inputs.iter().enumerate() {
                        buffer[output] += input * buffer[self.weights.start + i * inputs.len() + j];
                    }
                    buffer[output] = relu(buffer[output]);
                }
            }
            None => {
                for (i, output) in self.outputs.into_iter().enumerate() {
                    buffer[output] = buffer[self.biases.start + i];
                    for (j, input) in self.inputs.into_iter().enumerate() {
                        buffer[output] += buffer[input]
                            * buffer[self.weights.start
                                + i * (self.inputs.end - self.inputs.start)
                                + j];
                    }
                    buffer[output] = relu(buffer[output]);
                }
            }
        }
    }
}

fn fitness_function(inputs: &[f64], outputs: &[f64]) -> f64 {
    let correct = (inputs[0] as u8 ^ inputs[1] as u8) as f64;
    1.0 - (outputs[0] - correct).powi(2)
}

fn relu(n: f64) -> f64 {
    f64::max(0.0, n)
}

fn sigmoid(n: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-n))
}

fn tanh(n: f64) -> f64 {
    n.tanh()
}

fn main() {
    let mut population = Population::new(&TOPOLOGY, POPULATION_COUNT);
    evolution(&mut population);
    println!("\nFittest specimen demo:");
    for inputs in XOR_INPUTS_SET {
        let outputs = population.networks[0].infer(&inputs);
        let outputs = &population.networks[0].buffer[outputs.start..outputs.end];
        println!(
            "{} ^ {} = {} (fitness: {})",
            inputs[0],
            inputs[1],
            outputs[0],
            fitness_function(&inputs, outputs),
        );
    }
}

fn evolution(population: &mut Population) {
    for i in 0..TRIALS {
        population.mutate(
            ELITE_COUNT,
            MUTATION_RATE,
            MUTATION_STRENGTH,
            MUTATION_RANGE,
        );
        population.evaluate(&XOR_INPUTS_SET);
        population.select(ELITE_COUNT);

        println!(
            "Generation #{}'s highest fitness score: {}",
            i + 1,
            population.networks[0].fitness
        );
    }
}
