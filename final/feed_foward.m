function [net, pred] = feed_foward(input, net)
    %tanh function for activation
    function x = activation_function(input)        
        x = tanh(input);
    end
    function layer = dropout(input, dropoutRate)
        [sizeLayer, ~] = size(input);
        dropOutarr = ones(sizeLayer,1);
        for ii = 1:sizeLayer
            if rand < dropoutRate
            end
        end
        layer = input .* dropOutarr;
    end

    net.layer{1,1} = input;
    for index_layer = 2 : net.layer_num
          net.layer{index_layer-1, 1} = dropout(net.layer{index_layer-1,1}, 0.5);
          net.layer{index_layer, 1} = net.weight{index_layer, 1} * net.layer{index_layer-1, 1} + net.bias{index_layer, 1};
          net.layer{index_layer, 1} = activation_function(net.layer{index_layer, 1});
    end

    [~,ind] = max(net.layer{index_layer, 1});
    pred = zeros(size(net.layer{index_layer, 1}));
    for i=1:size(ind,2)
    pred(ind(i),i) = 1;
    end
end
