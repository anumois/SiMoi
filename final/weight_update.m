function net = weight_update(net, net_update, l_rate)
    % w <- w - learningRate * gradient
    for index_layer = 2 : net.layer_num
        net.weight{index_layer, 1} = net.weight{index_layer, 1} - (l_rate.learning_rate * net_update{index_layer, 1});
    end
       
end