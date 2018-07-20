clc
clear all

% LOAD THE DATA IN THE FORM OF VECTORS
load('x.mat')  
load('y.mat')
% y=[y y];
 
%ASSIGN THE DATA INTO REQUIRED VARIABLES
X=x(1:1000,:);
Y=y(1:1000,:);

% CLEAR PREVIOUS VARIABLES
clear x y

% REASSIGN THE DATA
x=X;
y=Y;

% DEFINE THE LEARNING LATE
L_rate=0.5;

% DEFINE THE NUMBER OF INPUT AND OUTPUT LAYERS
N_ip=length(x(1,:));
N_op=length(y(1,:));

% DEFINING THE NUMBER OF HIDDEN LAYERS
N_h_layers=1;


for i=1:1:N_h_layers

    N_neurons_h_layers(i)=N_ip-1;

end

% INITIALIZE WEIGHTS AND BIASES VARIABLES
W_ip=rand(N_ip,N_neurons_h_layers(1));
C_ip=rand(1,N_neurons_h_layers(1));

% INITIALIZE THE WEIGHTS MATRIX AND BIAS MATRIX WITH RANDOM NUMNERS
for i=1:1:N_h_layers
    
    W_h_layers(:,:,i)=rand(N_neurons_h_layers(1),N_op);
    C_h_layers(:,:,i)=rand(1,N_op);

end


% UPDATING THE WEIGHTS AND BIAS MATRIX VALUES
for iter=1:1:100000
    
a1=[ones(length(x(:,1)),1) x]*[C_ip; W_ip(:,:,1)];

z1=((1+exp(-a1)).^(-1));

S_dash_a1=z1.*(1-z1);

a2=[ones(length(z1(:,1)),1) z1]*[C_h_layers; W_h_layers(:,:,1)];

z2=((1+exp(-a2)).^(-1));

S_dash_a2=z2.*(1-z2);

J=sum((1/(2*length(x(:,1))))*sum((-y+z2).^2))

       
for j=1:1:length(y(1,:))
    for i=1:1:length(z1(1,:))         

        BW_h_layers(i,j)=sum(((1/(length(x(:,1)))).*(-y(:,j)+z2(:,j))).*z2(:,j).*(1-z2(:,j)).*z1(:,i));

    end
end
         
BW_ip=zeros(length(x(1,:)), length(z1(1,:)));

for i=1:1:length(z1(1,:))
    
    for j=1:1:length(x(1,:))
        
        for z=1:1:length(y(1,:))
            
            BW_ip(j,i)=BW_ip(j,i)+(1/(length(x(:,1))))*sum((z2(:,z)-y(:,z)).*S_dash_a2(:,z).*W_h_layers(i,z).*ones(length(z2(:,1)),1).*S_dash_a1(:,i).*x(:,j));
%             (1/(length(x(:,1))))*sum((z2(:,z)-y(:,z)).*S_dash_a2(:,z).*W_h_layers(i,z).*ones(length(S_dash_a2(:,1)),1).*S_dash_a1(:,i).*x(:,j));
        end
        
    end
    
end


for j=1:1:length(y(1,:))
%     for i=1:1:length(z1(1,:))         

        BC_h_layers(1,j)=sum(((1/(length(x(:,1)))).*(-y(:,j)+z2(:,j))).*S_dash_a2(:,j));

%     end
end
      
BC_ip=zeros(1, length(z1(1,:)));

for i=1:1:length(z1(1,:))
    
%     for j=1:1:length(x(1,:))
        
        for z=1:1:length(y(1,:))
            
            BC_ip(1,i)=BC_ip(1,i)+(1/(length(x(:,1))))*sum((z2(:,z)-y(:,z)).*S_dash_a2(:,z).*W_h_layers(i,z).*ones(length(z2(:,1)),1).*S_dash_a1(:,i));
%             (1/(length(x(:,1))))*sum((z2(:,z)-y(:,z)).*S_dash_a2(:,z).*W_h_layers(i,z).*ones(length(S_dash_a2(:,1)),1).*S_dash_a1(:,i).*x(:,j));
        end
        
%     end
    
end

        
% BW_ip=W;
C_h_layers=C_h_layers-L_rate*BC_h_layers;
C_ip=C_ip-L_rate*BC_ip;


W_h_layers=W_h_layers-L_rate*BW_h_layers;
W_ip=W_ip-L_rate*BW_ip;
clear BW_h_layers BC_h_layers BW_ip BC_ip

end

save('C_ip','C_ip')
save('W_ip','W_ip')

save('C_h_layers','C_h_layers')
save('W_h_layers','W_h_layers')