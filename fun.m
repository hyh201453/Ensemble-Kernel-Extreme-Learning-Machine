function  fitness = fun(x,p_train,T_train)   
    %% Obtain optimization parameters
    Regularization_coefficient = x(1);
    Kernel_para = x(2);
    Kernel_type = 'rbf';
    %% Cross-validation parameters
    num_folds = 5;
    num_size = size(p_train, 2);
    indices = crossvalind('Kfold', num_size, num_folds);
    error = zeros(1, num_folds);
    %% Cross-validation
    for i = 1:num_folds
        valid_data = (indices == i);  
        train_data = ~valid_data;    
        pv_train = p_train(:, train_data);  
        tv_train = T_train(:, train_data);  
        pv_valid = p_train(:, valid_data);  
        tv_valid = T_train(:, valid_data);  
        %% train stage
        [~, InputWeight] = kelmTrain(pv_train, tv_train, Regularization_coefficient, Kernel_type, Kernel_para);
        %% test stage
        t_sim = kelmPredict(pv_train, InputWeight, Kernel_type, Kernel_para, pv_valid);
        %% Calculate the error
        [~, T_sim] = max(t_sim, [], 1);
        [~, T_valid] = max(tv_valid, [], 1);
        error(i) = 1 - sum(T_sim == T_valid) / length(T_valid);
    end
    %% Calculate the mean error as the fitness
    fitness = mean(error);
end