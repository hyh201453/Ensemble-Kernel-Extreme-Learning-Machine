function [TestingTime,TestingAccuracy] = DT(train_data, test_data)
T=train_data(:,1)';                   
P=train_data(:,2:size(train_data,2))';
clear train_data;                     
TV.T=test_data(:,1)';                
TV.P=test_data(:,2:size(test_data,2))';
clear test_data;
model = fitctree(P',T');
start_time_test=cputime;
for i= 1:length(TV.P)
    predict_label(i) = predict(model, TV.P(:,i)');
end
end_time_test=cputime;
TestingTime=end_time_test-start_time_test;
TestingAccuracy = sum(predict_label == TV.T)/length(TV.T)*100;