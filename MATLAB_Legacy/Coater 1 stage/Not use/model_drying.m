function [dzdt] = model_drying(t,z,z_down,z_up,i)


global N_parameter n c r

global N_bed Volume_particle Surface_particle Length_particle Mass_particle Diameter_particle  ...
    Volume_stage Volume_bed Diameter_bed

global Gas_constant cp_p cp_dry_N2 cp_v DMC cp_c Temp_wb Molecular_weight_acetone ...
    Q_lat Volume_N2_stage rho_N2 Mass_N2 Molecular_weight_N2

global Flow_rate_N2 Flow_rate_coating Flow_rate_at absolute_humidity Temp_at Pressure ...
    Gas_moisture_0 Temp_N2_0

%%%%%%%%%%%%%%%%%%%%Calculating the model parameter%%%%%%%%%%%%%%%%%%%%%%%%


R_D=drying_rate(z(2),z(3),z(5),z(6),i);
cp_a=cp_dry_N2 + z(3)*cp_v;
alpha_p=heat_coef(cp_a);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%  NON COATING PART   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%% First stage case  i=1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if i==1

%Particle balance on a single control volume
dzdt(1)=0;

%Moisture (acetone) balance of the particle

dzdt(2)=( Mass_particle*N_bed*z_up(2)*r -R_D*Mass_particle*z(1) - Mass_particle*N_bed*r*z(2))/(Mass_particle*z(1));

%Moisture (acetone) balance in the gas phase
dzdt(3)=(Flow_rate_N2*(Gas_moisture_0-z(3)) + R_D*z(1)*Mass_particle)/Mass_N2;


%Coating  mass balance of the particle
%dzdt(4)=( Mass_particle*N_bed*z_up(4)*r - Mass_particle*N_bed*r*z(4) )/(Mass_particle*z(1));
dzdt(4)=0;
%Energy balance of the particles


dzdt(5)= ( Mass_particle*N_bed*cp_p*r*(z_up(5)-z(5))+ alpha_p*z(1)*Surface_particle*(z(6)-z(5)) ...
    - R_D*z(1)*Mass_particle*Q_lat ) / (z(1)*Mass_particle*cp_p);

%Energy balance of the gas phase

dzdt(6)= (Flow_rate_N2*cp_a*(Temp_N2_0-z(6)) - alpha_p*z(1)*Surface_particle*(z(6)-z(5)) - R_D*z(1)*Mass_particle*cp_v*(z(6)-z(5))...
    /(Mass_N2*cp_a) );

end







%%%%%%%%%%%%%%%%%%%% Non Coating stage case  1<i<n %%%%%%%%%%%%%%%%%%%%%%%%

if i>1 && i < n
    

%Particle balance on a single control volume

dzdt(1)= 0;

%Moisture (acetone) balance of the particle


dzdt(2)= (Mass_particle*N_bed*z_up(2)*r + Mass_particle*N_bed*z_down(2)*r - 2*Mass_particle*N_bed*z(2)*r - R_D*Mass_particle*z(1)) /(Mass_particle*z(1));

%Moisture (acetone) balance in the gas phase

dzdt(3)= (Flow_rate_N2*(z_down(3)-z(3)) + R_D*z(1)*Mass_particle)/Mass_N2;

%Coating  mass balance of the particle

%dzdt(4)= (Mass_particle*N_bed*z_down(4)*r + Mass_particle*N_bed*z_up(4)*r- 2*Mass_particle*N_bed*r*z(4) )/(Mass_particle*z(1));
dzdt(4)=0;
%Energy balance of the particles



dzdt(5)= ( Mass_particle*N_bed*cp_p*r*(z_up(5)-z(5)) + Mass_particle*N_bed*cp_p*r*(z_down(5)-z(5)) + alpha_p*z(1)*Surface_particle*(z(6)-z(5)) ...
    - R_D*z(1)*Mass_particle*Q_lat ) / (z(1)*Mass_particle*cp_p);

%Energy balance of he as phase

dzdt(6)= (Flow_rate_N2*cp_a*(z_down(6)-z(6)) - alpha_p*z(1)*Surface_particle*(z(6)-z(5)) - R_D*z(1)*Mass_particle*cp_v*(z(6)-z(5))...
    /(Mass_N2*cp_a) );

end



%%%%%%%%%%%%%%%%%%%%%%%%% Last stage case  i=n %%%%%%%%%%%%%%%%%%%%%%%%%%%%



if i==n
    
%Particle balance on a single control volume

dzdt(1)= 0;

%Moisture (acetone) balance of the particle


dzdt(2)= ( Mass_particle*N_bed*z_down(2)*r - Mass_particle*N_bed*z(2)*r - R_D*Mass_particle*z(1)) /(Mass_particle*z(1));

%Moisture (acetone) balance in the gas phase

dzdt(3)= (Flow_rate_N2*(z_down(3)-z(3)) + R_D*z(1)*Mass_particle)/Mass_N2;

%Coating  mass balance of the particle

%dzdt(4)= (Mass_particle*N_bed*z_down(4)*r - Mass_particle*N_bed*r*z(4))/(Mass_particle*z(1));
dzdt(4)=0;

%Energy balance of the particles

dzdt(5)= ( Mass_particle*N_bed*cp_p*r*(z_down(5)-z(5)) + alpha_p*z(1)*Surface_particle*(z(6)-z(5)) ...
    - R_D*z(1)*Mass_particle*Q_lat ) / (z(1)*Mass_particle*cp_p);

%Energy balance of the gas phase

dzdt(6)= (Flow_rate_N2*cp_a*(z_down(6)-z(6)) - alpha_p*z(1)*Surface_particle*(z(6)-z(5)) - R_D*z(1)*Mass_particle*cp_v*(z(6)-z(5))...
    /(Mass_N2*cp_a) );


end



dzdt=transpose(dzdt);


end