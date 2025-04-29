clear all
close all
clc
%% Whether to reduce features
remove_channel=0;
PCA_downscaling=0;
%% possible combinations of removed channels
if remove_channel==1
    remove_num=0;
    ch=1:10;
    combinations = nchoosek(ch, remove_num);
else
    combinations=0;
end
%% 
for remove_ch=1:size(combinations)
    Path = fullfile(pwd, 'example_data\');
    Files = dir(fullfile(Path, '*.mat'));
    num_sub=length(Files);
    all_emg_train = cell(1, num_sub);
    all_emg_test = cell(1, num_sub);
    %% train stage
    for subject=1:num_sub
        %% load data
        load([Path,Files(subject).name]);
        %% parameter setting
        stimulus=restimulus;
        repetition=rerepetition;
        deadzone=10^-5;
        winsize=25;
        wininc=5;
        %% remove channel
        if remove_channel==1
            emg(:,combinations(remove_ch,:))=[];
        end
        %% feature extraction
        [tmp_train,tmp_test]=Loader(emg,stimulus,repetition,deadzone,winsize,wininc);
        %% PCA downscaling
        if PCA_downscaling==1
            dimension=40;
            pca_trainData=tmp_train(:,2:end);
            pca_testData=tmp_test(:,2:end);
            [coeff, score, ~, ~, explained] = pca(pca_trainData);
            PCA_trainData = score(:, 1:dimension);
            PCA_testData = (pca_testData - mean(pca_trainData)) * coeff(:, 1:dimension);
            tmp_train(:,2:end)=[];
            tmp_test(:,2:end)=[];
            tmp_train=cat(2,tmp_train,PCA_trainData);
            tmp_test=cat(2,tmp_test,PCA_testData);
        end
        %% Integration of data
        all_emg_train{subject}=tmp_train;
        all_emg_test{subject}=tmp_test;
    end
    %% test stage
    for subject=1:num_sub
        %% select data
        emg_train=all_emg_train{1,subject};
        emg_test=all_emg_test{1,subject};
        %% Bayes
        [TestTime(subject,1),TestAccuracy(subject,1)] = Bayes(emg_train, emg_test);
        %% KNN
        K=5;
        [TestTime(subject,2), TestAccuracy(subject,2)] = KNN(emg_train, emg_test,K);
        %% LDA
        [TestTime(subject,3), TestAccuracy(subject,3)] = LDA(emg_train, emg_test);
        %% DT
        [TestTime(subject,4),TestAccuracy(subject,4)] = DT(emg_train, emg_test);
        %% BP
        [TestTime(subject,5), TestAccuracy(subject,5)] = BP(emg_train, emg_test);
        %% ELM
        ActivationFunction=1;
        NumberofHiddenNeurons=200;
        Elm_Type='sig';
        [TestTime(subject,6), TestAccuracy(subject,6)] = ELM(emg_train, emg_test,ActivationFunction, NumberofHiddenNeurons, Elm_Type);
        %% kELM
        Regularization_coefficient=1;
        Kernel_para=10;
        Kernel_type='RBF_kernel';
        [TestTime(subject,7), TestAccuracy(subject,7)] = kELM(emg_train, emg_test,Regularization_coefficient,Kernel_para,Kernel_type);
        %% ekELM
        Regularization_coefficient=[1,10];
        Kernel_para=[1,100];
        Kernel_type='RBF_kernel';
        [TestTime(subject,9),TestAccuracy(subject,8),TestAccuracy(subject,9)] = ekELM(emg_train,emg_test,Regularization_coefficient,Kernel_para,Kernel_type);
    end
    result{remove_ch}=struct('all_TestTime', TestTime, 'all_TestAccuracy', TestAccuracy);
end