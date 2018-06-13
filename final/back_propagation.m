function net_update = back_propagation(net, data_output)
    net_update = cell(size(net.weight));
    % accumulate gradient each step
    % dL/dOut = o - y
    dLdOut = net.layer{net.layer_num,1} - data_output;
    
    % index_layer 1 : chain rule for second-third layer, 2: chain rule for
    % first - second layer
    for index_layer = 1:net.layer_num -1
        % Derivative of activation function (tanh) = 1 - t^2, therefore
        % multiply 1 - o^2 to dLdOut to form gradient
        gradient = dLdOut .* (1 - (net.layer{net.layer_num,1}.^2));
        % Iterate only when index_layer > 1
        for i = 1: index_layer -1
            % dL/dout for inner layer
            gradient =  transpose((net.weight{net.layer_num-i+1,1})) * gradient;
            % dout/din
            gradient = gradient .* (1- (net.layer{net.layer_num-i,1}.^2));
        end
        
        % multiply hidden layer/input layer to the gradient as din/dw
        gradient = gradient * transpose((net.layer{(net.layer_num)-index_layer , 1}));
        net_update{net.layer_num+1-index_layer,1} = gradient; 
    end
    
end