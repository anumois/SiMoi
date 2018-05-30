function [valid_p] = validateNN(model,x_valid)
      x_test = model.coeff' * x_valid;
      [~,num_data_test] = size(x_test);
      valid_p = zeros(num_data_test,1);
      for index_data_test = 1 : num_data_test
            data_input = x_test(:,index_data_test);            
            [model, pred] = feed_foward(data_input, model);
            [~,result] = max(pred);            
            valid_p(index_data_test) = result;
      end
end

