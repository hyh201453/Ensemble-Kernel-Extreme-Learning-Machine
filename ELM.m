function [TestingTime,TestingAccuracy] = ELM(train_data, test_data, ActivationFunction, NumberofHiddenNeurons, Elm_Type)
[ELM_Model] = train_ELM(train_data,train_data,ActivationFunction, NumberofHiddenNeurons, Elm_Type);
[TestingTime, TestingAccuracy] = test_ELM(test_data, ELM_Model,ActivationFunction, Elm_Type);
end
