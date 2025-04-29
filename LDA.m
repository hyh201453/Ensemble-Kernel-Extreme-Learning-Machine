function [TestingTime, TestingAccuracy] = LDA(train_data, test_data)
T=train_data(:,1)';                   
P=train_data(:,2:size(train_data,2))';
clear train_data;                     
TV.T=test_data(:,1)';                  
TV.P=test_data(:,2:size(test_data,2))';
clear test_data;                     
obj = fitcdiscr(P',T','discrimType','pseudoLinear');
start_time_test=cputime;
pred_temp = predict(obj,TV.P');
end_time_test=cputime;
TestingTime=end_time_test-start_time_test;
TestingAccuracy = sum(pred_temp == TV.T')/length(TV.T)*100;