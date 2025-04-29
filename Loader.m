function [emg_train,emg_test]=Loader(emg,stimulus,repetition,deadzone,winsize,wininc)
% extract feature
feature=[];
[feat, featStim, featRep] = ParFeatureExtractor(emg,stimulus,repetition,deadzone,winsize,wininc,'getmavfeat');
feature=[feature feat];
[feat, featStim, featRep] = ParFeatureExtractor(emg,stimulus,repetition,deadzone,winsize,wininc,'getsscfeat');
feature=[feature feat];
[feat, featStim, featRep] = ParFeatureExtractor(emg,stimulus,repetition,deadzone,winsize,wininc,'getwlfeat');
feature=[feature feat];
[feat, featStim, featRep] = ParFeatureExtractor(emg,stimulus,repetition,deadzone,winsize,wininc,'getrmsfeat');
feature=[feature feat];
[feat, featStim, featRep] = ParFeatureExtractor(emg,stimulus,repetition,deadzone,winsize,wininc,'getiavfeat');
feature=[feature feat];
feature=[featStim feature];
feature=sortrows(feature,1);
% select train/test data
train_num=100;
test_num=20;
unique_vals = unique(feature(:, 1));
emg_train = [];
emg_test = [];
for i = 1:length(unique_vals)
    current_val_rows = feature(feature(:, 1) == unique_vals(i), :);
    randomNumbers1 = randperm(size(current_val_rows, 1), train_num);
    selected_rows = current_val_rows(randomNumbers1, :);
    current_val_rows(randomNumbers1, :)=[];
    randomNumbers2 = randperm(size(current_val_rows, 1), test_num);
    remaining_rows= current_val_rows(randomNumbers2, :);
    emg_train = [emg_train; selected_rows];
    emg_test = [emg_test; remaining_rows];
end
emg_train(:,1)=emg_train(:,1)+1;
emg_test(:,1)=emg_test(:,1)+1;
emg_train=double(emg_train);
emg_test=double(emg_test);
end