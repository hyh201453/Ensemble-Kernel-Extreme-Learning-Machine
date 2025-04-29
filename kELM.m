function [TestingTime,TestingAccuracy] = kELM(train_data, test_data,Regularization_coefficient,Kernel_para,Kernel_type)
[Weight1] = train_kELM(Regularization_coefficient,Kernel_para,train_data,train_data,Kernel_type);
[TestingTime,~,TestingAccuracy] = test_kELM(train_data,test_data,Weight1,Kernel_type,Kernel_para);
end