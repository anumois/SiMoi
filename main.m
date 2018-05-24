clear; clc;
%% your parameter here%%
%
%% algorithm %%
[x_train,y_train,x_valid,y_valid] = createDataset('train_feat.csv', 'train_label.csv','valid_feat.csv', 'valid_label.csv');
a = redistribute(y_train);
[coeff,score,latent]=pca(x_train(:,a)');
latent_por=latent/sum(latent);
coeff=coeff(:,1:find(latent_por<0.05,1));
score=score(:,1:find(latent_por<0.05,1));
%% training 
model= algorithm(x_train, y_train);

%% validation 
[valid_p] = validation(model, x_valid);

%% your analysis here 


%% test 
[x_test, y_test] = createDatasetTest('test_feat.csv', 'test_label.csv');
[test_p] = validation(model, x_test);

%% Find Accuracy
disp('class predict')
disp([y_valid valid_p]);
disp([y_test test_p]);
valid_acc =mean(y_valid== valid_p)*100;
test_acc =mean(y_test == test_p)*100;
fprintf('\valid_acc =%d\n',valid_acc)
fprintf('\test_acc =%d\n',test_acc)