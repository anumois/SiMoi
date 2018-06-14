clear; clc;
%% your parameter here%%
l=1; %slack value
EpL = 1200; % Number of minimum data
cri=10; % For the fast code validity tests. cri means y_train = 1~cri data are used.
% Decision structre is also exist as hyper parameter
%% algorithm %%
[x_train,y_train,x_valid,y_valid] = createDataset('train_feat.csv', 'train_label.csv','valid_feat.csv', 'valid_label.csv');
[x_test] = csvread('test_feat.csv');
x_train=x_train(:,y_train<cri+1);
y_train=y_train(y_train<cri+1);
x_valid=x_valid(:,y_valid<cri+1);
y_valid=y_valid(y_valid<cri+1);
a = redistribute(y_train,EpL,cri);
%% training 
model= algorithmNN(x_train(:,a)', y_train(a),cri, x_valid, y_valid);
%% validation 
[valid_p] = validateNN(model, x_valid);
[test_p] = validateNN(model, x_test);
csvwrite('testOut.csv',test_p);
%% Find Accuracy
valid_acc =mean(y_valid== valid_p)*100;
fprintf('valid_acc =%f\n',valid_acc)

%% Return confusion matrix
 figure();
 imagesc(confusion_generator(y_valid,valid_p)); 
 colorbar;