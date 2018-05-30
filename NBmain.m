clear; clc;
%% your parameter here%%
l=1; %slack value
EpL = 200; % Number of minimum data
T=15000; % number of iterations
sig = 0.5; % Size of kernel
cri=10; % For the fast code validity tests. cri means y_train = 1~cri data are used.
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
PCA_score = PCA_score';
model= algorithmNaiveBayes(x_train(:,a), y_train(a), cri,PCA_comp);
%model= algorithm_kernelSVM_slow(PCA_comp, PCA_score, y_train(a),T,l,sig);
%% validation 
[valid_p] = validation_NaiveBayes(model, x_valid, cri);

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