function [Model] = train_ELM(train_data,valid_data, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
REGRESSION=0;
CLASSIFIER=1;
T=train_data(:,1)';                   
P=train_data(:,2:size(train_data,2))';
clear train_data;
TV.T=valid_data(:,1)';                
TV.P=valid_data(:,2:size(valid_data,2))';
clear valid_data;                           
NumberofTrainingData=size(P,2);        
NumberofTestingData=size(TV.P,2);     
NumberofInputNeurons=size(P,1);
if Elm_Type~=REGRESSION
    sorted_target=sort(cat(2,T,TV.T),2);
    label=zeros(1,1);                   
    label(1,1)=sorted_target(1,1);
    j=1;
    for i = 2:(NumberofTrainingData+NumberofTestingData)
        if sorted_target(1,i) ~= label(1,j)
            j=j+1;
            label(1,j) = sorted_target(1,i);
        end
    end
    number_class=j;                  
    NumberofOutputNeurons=number_class;
    temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
    for i = 1:NumberofTrainingData
        for j = 1:number_class
            if label(1,j) == T(1,i)
                break; 
            end
        end
        temp_T(j,i)=1;              
    end
    T=temp_T*2-1;                     
    temp_TV_T=zeros(NumberofOutputNeurons, NumberofTestingData);
    for i = 1:NumberofTestingData
        for j = 1:number_class
            if label(1,j) == TV.T(1,i)
                break; 
            end
        end
        temp_TV_T(j,i)=1;            
    end
    TV.T=temp_TV_T*2-1;              
end
InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1; 
BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);              
tempH=InputWeight*P;
clear P;  
ind=ones(1,NumberofTrainingData);    
BiasMatrix=BiasofHiddenNeurons(:,ind);
tempH=tempH+BiasMatrix;
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        H = 1 ./ (1 + exp(-tempH));
    case {'sin','sine'}
        H = sin(tempH);           
    case {'hardlim'}
        H = double(hardlim(tempH));
    case {'tribas'}
        H = tribas(tempH);        
    case {'radbas'}
        H = radbas(tempH); 
end
clear tempH;
OutputWeight=pinv(H') * T';
Y=(H' * OutputWeight)';
if Elm_Type == REGRESSION 
    TrainAccuracy=sqrt(mse(T - Y)); 
end
clear H;
start_time_test=cputime;
tempH_test=InputWeight*TV.P;
clear TV.P;
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);
tempH_test=tempH_test + BiasMatrix;
switch lower(ActivationFunction)
    case {'sig','sigmoid'}  
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'sin','sine'}
        H_test = sin(tempH_test);        
    case {'hardlim'}    
        H_test = hardlim(tempH_test);        
    case {'tribas'}      
         H_test = tribas(tempH_test);        
    case {'radbas'}     
         H_test = radbas(tempH_test);                     
end
TY=(H_test' * OutputWeight)';
end_time_test=cputime;
ValidTime=end_time_test-start_time_test;
if Elm_Type == REGRESSION
    ValidAccuracy=sqrt(mse(TV.T - TY));
end
if Elm_Type == CLASSIFIER 
    MissClassificationRate_Training=0;
    MissClassificationRate_Testing=0;
    for i = 1 : size(T, 2) 
        [x, label_index_expected]=max(T(:,i));
        [x, label_index_actual]=max(Y(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Training=MissClassificationRate_Training+1;
        end
    end
    TrainAccuracy=1-MissClassificationRate_Training/size(T,2);
    for i = 1 : size(TV.T, 2)
        [x, label_index_expected]=max(TV.T(:,i));
        [x, label_index_actual]=max(TY(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Testing=MissClassificationRate_Testing+1;
        end
    end
    ValidAccuracy=1-MissClassificationRate_Testing/size(TV.T,2);
end
Model=struct('InputWeight', InputWeight, 'BiasofHiddenNeurons', BiasofHiddenNeurons, 'OutputWeight', OutputWeight);