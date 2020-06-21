%author: Rui Wang
%date: 2018
%Copyright@ JNU_B411
%department: school of artificial intelligence and computer science

clear;
close all;
clc;

%step1 upload .mat data
%load ImgData_HE_ETH_100_1
% load ETH_80_Data_New
load demo-ETH
t_star = cputime; % 

%step2: add label to the training data and test data
Train_lables = zeros(1,40);
Test_lables = zeros(1,40);

% step2
l=5;
k=l;
a=linspace(1,8,8);
i1=1;
while(k<=40)
    while(i1<=8)
        for i=1:8
            i_train=l*(i-1)+1;
            Train_lables(i_train:k)=a(i1);
            k=k+5;
            i1=i1+1;
        end
    end
end

% step3
l1=5;
k1=l1; 
a1=linspace(1,8,8);
i2=1;
while(k1<=40)
    while(i2<=8)
        for i=1:8
            i_test=l1*(i-1)+1;
            Test_lables(i_test:k1)=a(i2);
            k1=k1+5;
            i2=i2+1;
        end
    end
end

%step5:
param.d = 400;%
d = param.d;
basis = eye(d);%
% cov_train_disturb=cell(1,40);
ImgData_HE_train = cell(1,40);
ImgData_HE_test = cell(1,40);
accuracy_matrix = zeros(1,10); %

for iteration = 1 : 1
    
    log_cov_train_Gras = cell(1,40); %
    log_cov_train_Spd = cell(1,40); %
    
    % cov_test_disturb=cell(1,32);
    log_cov_test_Gras=cell(1,40);
    log_cov_test_Spd=cell(1,40);
    
    ImgData_HE_train = ETH_train; % All_ImgData_HE_train{iteration};
    ImgData_HE_test = ETH_test; % All_ImgData_HE_test{iteration};
    
    tic
    [ls_train, q1] = compute_sub(ImgData_HE_train); %
    cov_train = compute_cov(ImgData_HE_train);
    t_star_train1=cputime;
    
for i=1:40
    
    temp_tr_Gras=ls_train{i};
    temp_tr_Spd=cov_train{i};
    log_cov_train_Spd{i}=logm(temp_tr_Spd); % 
    log_cov_train_Gras{i}=temp_tr_Gras; % 
    
end
    toc
    disp('obtaining traing data')
    %clear center_matrix_train;
    t_train1= cputime - t_star_train1; %
    % clear cputime;
    t_star_test1=cputime; % 计时用;
    %step6:
    tic
    [ls_test, q2] = compute_sub(ImgData_HE_test); 
    cov_test = compute_cov(ImgData_HE_test);
    
for i=1:40
    temp_te_Gras = ls_test{i};
    temp_te_Spd = cov_test{i};
    log_cov_test_Gras{i} = temp_te_Gras; %
    log_cov_test_Spd{i} = logm(temp_te_Spd); %
end
    toc
    disp('obtaing test data')
    t_test1= cputime - t_star_test1;
    % clear cputime;
    
    %step7: building the kernel matrices for the Grassmannian data and SPD manifold-valued data, respectively
    kmatrix_train=zeros(size(log_cov_train_Gras,2),size(log_cov_train_Gras,2)); % 
    kmatrix_test=zeros(size(log_cov_train_Gras,2),size(log_cov_test_Gras,2)); % 
    
    kmatrix_train_Spd=zeros(size(log_cov_train_Spd,2),size(log_cov_train_Spd,2)); % 
    kmatrix_test_Spd=zeros(size(log_cov_train_Spd,2),size(log_cov_test_Spd,2)); % 
    
    t_star_train2=cputime;
    
    tic
for i=1:size(log_cov_train_Gras,2)
    for j=1:size(log_cov_train_Gras,2)
        cov_i_Train=log_cov_train_Gras{i};% cov_i_Train is actually the log-mapped cov
        cov_j_Train=log_cov_train_Gras{j};% cov_i_Train is actually the log-mapped cov
        temp_i = cov_i_Train * cov_i_Train';
        temp_j = cov_j_Train*cov_j_Train';
        temp_i = temp_i(:);
        temp_j = temp_j(:);
        kmatrix_train(i,j) = temp_i' * temp_j; % trace((cov_i_Train*cov_i_Train')*(cov_j_Train*cov_j_Train'));%141*141
        kmatrix_train(j,i)=kmatrix_train(i,j);
    end
end
toc
disp('train kernel Grass')


tic
for i=1:size(log_cov_train_Spd,2)
    for j=1:size(log_cov_train_Spd,2)
        cov_i_Train=log_cov_train_Spd{i}; % cov_i_Train is actually the log-mapped cov
        cov_j_Train=log_cov_train_Spd{j}; % cov_i_Train is actually the log-mapped cov
        cov_i_Train_reshape=reshape(cov_i_Train,size(cov_i_Train,1)*size(cov_i_Train,2),1);
        cov_j_Train_reshape=reshape(cov_j_Train,size(cov_j_Train,1)*size(cov_j_Train,2),1);
        kmatrix_train_Spd(i,j)=cov_i_Train_reshape'*cov_j_Train_reshape;%141*141
        kmatrix_train_Spd(j,i)=kmatrix_train_Spd(i,j);
    end
end
toc
disp('train kernel SPD')

t_train2=cputime-t_star_train2;
% clear cputime;
t_star_test2=cputime;

tic
for i=1:size(log_cov_train_Gras,2)
    for j=1:size(log_cov_test_Gras,2)
        cov_i_Train=log_cov_train_Gras{i}; 
        cov_j_Test=log_cov_test_Gras{j}; 
        temp_i = cov_i_Train * cov_i_Train';
        temp_j = cov_j_Test*cov_j_Test';
        temp_i = temp_i(:);
        temp_j = temp_j(:);
        kmatrix_test(i,j) = temp_i' * temp_j; % trace((cov_i_Train*cov_i_Train')*(cov_j_Test*cov_j_Test'));%240*141
    end
end
toc
disp('test kernel Grass')

tic
for i=1:size(log_cov_train_Spd,2)
    for j=1:size(log_cov_test_Spd,2)
        cov_i_Train=log_cov_train_Spd{i};% cov_i_Train is actually the log-mapped cov
        cov_j_Test=log_cov_test_Spd{j};% cov_i_Train is actually the log-mapped cov
        cov_i_Train_reshape=reshape(cov_i_Train,size(cov_i_Train,1)*size(cov_i_Train,2),1);%拉成一个高纬的列向量
        cov_j_Test_reshape=reshape(cov_j_Test,size(cov_j_Test,1)*size(cov_j_Test,2),1);%拉成一个高纬的列向量
        kmatrix_test_Spd(i,j)=cov_i_Train_reshape'*cov_j_Test_reshape;%240*141
    end
end
toc
disp('test kernel SPD')

% two parameters for these two models
lamda1 = 0.8;  % for Grasmann kernel feature
lamda2 = 0.2; % for Spd kernel feature
alpha = 0.2; % balance parameter of the objective function

% Compute the core matrix U
tic
U = compute_metric_learning(kmatrix_train, kmatrix_train_Spd, lamda1, lamda2, Train_lables);
toc
disp('mmml')
dist = zeros(size(Train_lables,2),size(Test_lables,2)); % distance matrix

%% classification
tic
for i_dist=1:size(Train_lables,2)
        Y_train_spd = kmatrix_train_Spd(:,i_dist); 
        Y_train_gras = kmatrix_train(:,i_dist); 
     for j_dist=1:size(Test_lables,2)
        Y_test_spd = kmatrix_test_Spd(:,j_dist); 
        Y_test_gras = kmatrix_test(:,j_dist); 
        Y_dist1 = lamda1*(Y_train_gras-Y_test_gras)' * U * U' * (Y_train_gras-Y_test_gras) * lamda1;
        Y_dist2 = lamda2*(Y_train_spd-Y_test_spd)' * U * U' * (Y_train_spd-Y_test_spd) * lamda2;
        dist(i_dist,j_dist) = Y_dist1 + Y_dist2;
     end
end
 toc
 disp('classification')
 
 test_num=size(Test_lables,2); % number of test samples
 [dist_sort,index] = sort(dist,1,'ascend');
 %right_num=length(find((Test_labels'-Train_labels'(index(1,:)))==0));
 right_num = length(find((Test_lables'-Train_lables(index(1,:))')==0)); 
 accuracy = right_num/test_num;
 accuracy_matrix(iteration) = accuracy * 100;
 fprintf(1,'the number of right predicted samples of the %d-th combination is：%d %d\n',iteration,right_num );
 fprintf(1,'the classification score of the %d-th combination is: %d %d\n', iteration ,accuracy*100);

end
mean_accuracy=sum(accuracy_matrix) / 1.0;
fprintf(1,'mean classification score is: %d\n',mean_accuracy);
fprintf(1,'standard derivation is: %d\n',std(accuracy_matrix));
