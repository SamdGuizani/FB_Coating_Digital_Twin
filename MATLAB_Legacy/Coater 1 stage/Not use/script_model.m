
clc;
clear all;

N_parameter = 6; %(/)
N_bed=10000;  %(/)  %Nombre de particule dans le lit
Batch_size=6;%(kg)
n=10;
c=2;
r=0.5;
tspan=[0 2000];
R_D=ones(n);

Mass_particle=Batch_size / N_bed; %(kg)
Mass_N2=1;

% Paramètres d'entrée de procédés

Temp_particle_0=293.15; %(K)
Temp_N2_0=303.15; %(K)
Coating_mass_0=0; %(/)
Gas_moisture_0=0; % (/)
Particle_moisture_0=0; % (/)
Ni_1=N_bed/n;
Flow_rate_N2=10; %(kg/s)

zi=zeros(n,N_parameter); %Initialisation des paramètres avec les conditions d'entrée
zi_1= [Ni_1 Particle_moisture_0 Gas_moisture_0 Coating_mass_0 Temp_particle_0 Temp_N2_0];

zi(1,:)=zi_1;

dzidt=zeros(n,N_parameter);
for i=2:n-1
    
% Particle balance on a single control volume

dzidt(i,1)= 10;


% Moisture (acetone) balance of the particle

dzidt(i,2)= (N_bed*zi(i+1,2)*r + N_bed*zi(i-1,2)*r - N_bed*2*zi(i,2)*r - R_D(i)*zi(i,1))   *(1/10);

% Moisture (acetone) balance in the gas phase

dzidt(i,3)= (Flow_rate_N2*(zi(i-1,3)-zi(i,3)) + R_D(i)*zi(i,1)*Mass_particle)/Mass_N2;

% Coating  mass balance of the particle

dzidt(i,4)= 0;

% Energy balance of the particles;

dzidt(i,5)=

end

