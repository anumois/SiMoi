function net = initialize_network(num_neuron, init)
%% initialize structure
net.layer_num = length(num_neuron);
net.num_neuron = num_neuron;

%% initialize each layer
net.layer = cell(net.layer_num,1);
for layer_index = 1 : net.layer_num
      net.layer{layer_index, 1} = zeros(net.num_neuron(layer_index, 1), 1); 
end
%% initialize weight
net.weight = cell(net.layer_num,1);
for layer_index = 2 : net.layer_num
      net.weight{layer_index, 1} = init.weight_std * randn(net.num_neuron(layer_index, 1), net.num_neuron(layer_index-1, 1));
end
%% initialize bias
net.bias = cell(net.layer_num, 1);
for layer_index = 2 : net.layer_num
      net.bias{layer_index, 1} = init.bias_std * randn(net.num_neuron(layer_index, 1), 1);
end

end

