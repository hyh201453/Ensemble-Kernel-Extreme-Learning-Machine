function [TestTime_all,single_TestAccuracy,ensemble_TestAccuracy] = ekELM(emg_train,emg_test,Regularization_coefficient,Kernel_para,Kernel_type)
addpath path_kelm
% number of sub-KELM
ensemble_num=10;
all_result=[];
% eKELM
parfor i=1:ensemble_num
    [m, ~] = size(emg_train);
    randIdx = randperm(m);
    splitIndex = round(0.8 * m);
    train_data = emg_train(randIdx(1:splitIndex), :);
    valid_data= emg_train(randIdx(splitIndex+1:end), :);
    P_train = train_data(:,2:size(train_data,2))';
    T_train = train_data(:,1)';
    P_test=valid_data(:,2:size(valid_data,2))';
    T_test=valid_data(:,1)';
    t_train = ind2vec(T_train);
    t_test  = ind2vec(T_test);
    M = size(P_train, 2);
    N = size(P_test, 2);
    % CPO-KELM Model
    pop = 50;             % initial population
    Max_time = 100;        % maximum iteration number
    dim = 2;              % number of hyperparameters
    fobj = @(x) fun(x, P_train, t_train);
    [~, Best_pos, ~] = CPO(pop, Max_time, Regularization_coefficient, Kernel_para, dim, fobj); % 优化开始
    % train stage
    [Weight] = train_kELM(Best_pos(1),Best_pos(2),train_data,valid_data,'RBF_kernel');
    % test stage
    [TestTime(i),label_index_actual(i,:),TestAccuracy(i)] = test_kELM(train_data,emg_test,Weight,'RBF_kernel',Best_pos(2));
    all_result(i,:)=label_index_actual(i,:);
end
TestTime1=sum(TestTime);
single_TestAccuracy=mean(TestAccuracy);
TV.T = emg_test(:,1)';
start_time_test=cputime;
for i = 1 : size(TV.T, 2)
    ensemble_index_actual(i)=mode(all_result(:,i));
end
end_time_test=cputime;
TestTime2=end_time_test-start_time_test;
TestTime_all=TestTime1+TestTime2;
label_index_expected=TV.T ;
ensemble_TestAccuracy=mean(label_index_expected==ensemble_index_actual);
ensemble_TestAccuracy=ensemble_TestAccuracy*100;
end