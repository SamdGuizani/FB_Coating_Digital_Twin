function [dzdt] = model(t,z,z_down,z_up,i)


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


%1
dzdt(1)=0;

dzdt(2)= ( (1-DMC)*Flow_rate_coatingv - R_D*Mass_particle*z(1) ) / (Mass_particle*z(1));

dzdt(3)=


dzdt=transpose(dzdt);


end



