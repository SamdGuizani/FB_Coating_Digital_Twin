function alpha=Calcul_MassTransfer(part)




global Diameter_eq Diameter_bed

global rho_air Process_parameter Flow_rate_at


if part==1
    Flow_rate_air=Process_parameter(1,1);
 
elseif part ==2
    Flow_rate_air=Process_parameter(1,2)+Flow_rate_at; %(kg/s)

elseif part ==3
    Flow_rate_air=Process_parameter(1,3);
end


Section_bed=pi*0.25*Diameter_bed^2; %(m2)
Velocity_air=Flow_rate_air/(Section_bed*rho_air); %(m/s)

Viscosity_air= 1.94*1e-5; %(Pa*s)
Molecular_Diff_acetone=1e-6; %(m^2/s) %%%A MODIFIER %%%%%


Schmidt= Viscosity_air/(rho_air*Molecular_Diff_acetone);
Reynolds_particle=rho_air*Velocity_air*Diameter_eq/Viscosity_air;


Sherwood_particle= 2 + Schmidt^(2/5) *(0.43*Reynolds_particle^0.5 + 0.06*Reynolds_particle^(2/3));

alpha=Sherwood_particle*Molecular_Diff_acetone/Diameter_eq;

end