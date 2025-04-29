
function [Destination_fitness,bestPositions,Convergence_curve]=ismaforkelm(N,Max_iter,lb,ub,dim,fobj)

% ��Ⱥ��ʼ��
bestPositions=zeros(1,dim);
Destination_fitness=inf;%���ó�ʼȫ������Ӧ��
AllFitness = inf*ones(N,1);%��ʼ������Ⱥ����Ӧ��
weight = ones(N,dim);%ÿһ������Ȩ��
%��Ⱥ��ʼ��
lb = [1,1];%�±߽�
ub = [10,10];%�ϱ߽�
Ub=ub(1);Lb=lb(1);
X=Goodnode_initialization(N,dim,Ub,Lb);
it=1;  %��ʼ��������
lb=ones(1,dim).*lb; % �½�
ub=ones(1,dim).*ub; % �Ͻ�
z=0.03; % ��ʼ����z

% ��ѭ��
while  it <= Max_iter
    it
    for i=1:N
        % ����Ƿ��ڷ�Χ��
        Flag4ub=X(i,:)>ub;
        Flag4lb=X(i,:)<lb;
        X(i,:)=(X(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        AllFitness(i) = fobj(X(i,:));
    end
    [X,AllFitness]=ODE(X,AllFitness,ub,lb,fobj);
    
    if AllFitness(1) < Destination_fitness
        bestPositions=X(1,:);
        Destination_fitness = AllFitness(1);
    end

    [SmellOrder,SmellIndex] = sort(AllFitness);  %%ɸѡ�����ź�������Ⱥ
    worstFitness = SmellOrder(N);
    bestFitness = SmellOrder(1);
%     bestPositions=X(SmellIndex(1),:);
%     Destination_fitness = AllFitness(SmellIndex(1));
    
    S=bestFitness-worstFitness+eps; %�����ĸΪ0�Ĳ���
    %����ÿһ������Ȩ��
    for i=1:N
        for j=1:dim
            if i<=(N/2)  %�ο�Դ�룬ʽ2.5
                weight(SmellIndex(i),j) = 1+rand()*log10((bestFitness-SmellOrder(i))/(S)+1);
            else
                weight(SmellIndex(i),j) = 1-rand()*log10((bestFitness-SmellOrder(i))/(S)+1);
            end
        end
    end
    
    %���µ�ǰ���ŵ���Ⱥ����Ӧ��
    if bestFitness < Destination_fitness
        bestPositions=X(SmellIndex(1),:);
        Destination_fitness = bestFitness;
    end
    
    a = atanh(-(it/Max_iter)+1);   %�ο�ԭʽ2.4
    b = 1-it/Max_iter;
    % ����ÿһ����Ⱥ��λ��
    for i=1:N
        if rand<z     %Eq.(2.7)
            X(i,:) = (ub-lb)*rand+lb;
        else
            p =tanh(abs(AllFitness(i)-Destination_fitness));  %Eq.(2.2)
            vb = unifrnd(-a,a,1,dim);  %Eq.(2.3)
            vc = unifrnd(-b,b,1,dim);
            for j=1:dim
                r = rand();
                A = randi([1,N]);  % two positions randomly selected from population
                B = randi([1,N]);
                if r<p    %Eq.(2.1)
                    X(i,j) = bestPositions(j)+ vb(j)*(weight(i,j)*X(A,j)-X(B,j));
                else
                    X(i,j) = vc(j)*X(i,j);
                end
            end
        end
    end
    f_ave=mean(AllFitness);
    for g=1:N
        if fobj(X(i,:))<f_ave
            X(i,:)=X(i,:).*trnd(it);
            X(i,:) = Bounds(X(i,:), lb, ub);
        else
            rand_num=randperm(0.5*N,1);
            X(i,:)=(X(i,:)+X(rand_num,:))/2;
            X(i,:) = Bounds(X(i,:), lb, ub);
        end
    end
    
    Convergence_curve1(it)=Destination_fitness;
    it=it+1;
end


