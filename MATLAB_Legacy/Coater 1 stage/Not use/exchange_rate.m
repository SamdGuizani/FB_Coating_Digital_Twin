function r=exchange_rate()

global N_parameter n c r

global N_bed Volume_particle Surface_particle Length_particle Mass_particle Diameter_particle ... 
    Volume_stage Volume_bed Diameter_bed rho_particle Batch_size eps_mf

global Gas_constant cp_p cp_dry_N2 cp_v DMC cp_c Temp_wb Molecular_weight_acetone ...
    Q_lat Volume_N2_stage rho_N2 Mass_N2 Molecular_weight_N2

global Flow_rate_N2 Flow_rate_coating Flow_rate_at absolute_humidity Temp_at Pressure ...
    Gas_moisture_0 Temp_N2_0


Section_bed=pi*0.25*Diameter_bed^2;
Velocity_N2=Flow_rate_N2/(Section_bed*rho_N2) %(m3/s)
g=9.81; %(m/s^2)
Viscosity_N2= 1.6629*1e-5; %(Pa*s)

Heigth_mf= Batch_size/rho_particle * (1+eps_mf) / Section_bed %(m)

Archimede = Diameter_particle^3 *rho_N2*(rho_particle-rho_N2)*g/(Viscosity_N2^2);

Velocity_mf=Viscosity_N2/(rho_N2*Diameter_particle) * ((33.7^2 + 0.04084*Archimede ...
    )^0.5 - 33.7)

Velocity_bu=1; %%%%%% A MODIFIER %%%%%%%%%%%

r= 0.6 * ((Velocity_N2-Velocity_mf) * (1- (Velocity_N2-Velocity_mf)/Velocity_bu)) / Heigth_mf

end

