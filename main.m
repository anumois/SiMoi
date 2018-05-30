clear; clc;
%% your parameter here%%
l=1; %slack value
EpL = 200; % Number of minimum data
T=15000; % number of iterations
sig = 0.5; % Size of kernel
cri=10; % For the fast code validity tests. cri means y_train = 1~cri data are used.
k=5; %knn neighbors
% Decision structre is also exist as hyper parameter
%% algorithm %%
[x_train,y_train,x_valid,y_valid] = createDataset('train_feat.csv', 'train_label.csv','valid_feat.csv', 'valid_label.csv');
x_train=x_train(:,y_train<cri+1);
y_train=y_train(y_train<cri+1);
x_valid=x_valid(:,y_valid<cri+1);
y_valid=y_valid(y_valid<cri+1);
a = redistribute(y_train,EpL,cri);
[PCA_comp, PCA_score]=PCA_part(x_train(:,a));
%% training 
%model= algorithm_kernelSVM_quick(PCA_comp, PCA_score, y_train(a),T,l,sig);
%model= algorithm_kernelSVM_slow(PCA_comp, PCA_score, y_train(a),T,l,sig);
valid_p= algorithm_validation_KNN(x_train, y_train,x_valid,EpL,cri,k,500); % 78 to 81%
%% validation 
%[valid_p] = validation_kernelSVM(model, x_valid, x_train(:,a),sig);
%% your analysis here 


%% test 
%[x_test, y_test] = createDatasetTest('test_feat.csv', 'test_label.csv');
%[test_p] = validation(model, x_test);

%% Find Accuracy
disp('class predict')
disp([y_valid valid_p]);
%disp([y_test test_p]);
valid_acc =mean(y_valid== valid_p)*100;
%test_acc =mean(y_test == test_p)*100;
fprintf('valid_acc =%f\n',valid_acc)
%fprintf('test_acc =%f\n',test_acc)

%% Return confusion matrix
% figure();
% imagesc(confusion_generator(y_valid,valid_p));
mat2gray(confusion_generator(y_valid,valid_p));