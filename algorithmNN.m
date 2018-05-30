function [model] = algorithmNN(x_train, y_train, cri, comp)
    %% data & neuron number setting    
    num_data_train = size(x_train, 1);    
    num_neuron_input = size(x_train, 2);
    num_neuron_output = cri;
    
    % Learning parameters    
    batch_size = 1;
    num_neuron_hidden = 47;
    % weight initialization setting
    init.weight_std = 0.1; % stdev of weight paramters
    init.bias_std = 0.1; % stdev of bias paramters
    % training setting
    training.num_epoch = 5; % num of epochs
    training.learning_rate = 0.005; % learning rate
    training.test_period = 100; % test every iteration
    
    % Initializations %
    training.current_epoch = 0;
    num_neuron = [num_neuron_input; num_neuron_hidden; num_neuron_output]; 
    training.num_input_graph = training.num_epoch * floor(num_data_train / training.test_period);
    training.graph = zeros(training.num_epoch, 1);
    training.index_graph = 0;
    
    %% initialize modelwork %%
    model = initialize_network(num_neuron, init);
    model.coeff = comp;
    k = 0;
    
        %% Training %%
    for epoch = 1 : training.num_epoch
          training.current_epoch = epoch;
          order_index_train = randperm(num_data_train);
          for index_data = 1 :batch_size: num_data_train
                data_input = x_train(order_index_train(index_data:index_data+batch_size-1), :)';
                data_output = -ones(cri,1);
                data_output(y_train(order_index_train(index_data:index_data+batch_size-1), :)) = 1;
                %% Forward computations
                [model,~] = feed_foward(data_input, model);
                %% Backward computations
                model_update = back_propagation(model, data_output);
                %% Weight update
                model = weight_update(model, model_update, training);
                k = k+1;
                % test step
                if mod(k, training.test_period) == 0 % Test period
                      disp(num2str(k))
                end 
          end
    end
end

