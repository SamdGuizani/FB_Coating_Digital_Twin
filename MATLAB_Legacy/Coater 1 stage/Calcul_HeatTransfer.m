function alpha=Calcul_HeatTransfer(cp_a,part)



global  Diameter_eq Diameter_bed

global  rho_air Flow_rate_at

global Process_parameter 




if part==1
    Flow_rate_air=Process_parameter(1,1);
 
elseif part ==2
    Flow_rate_air=Process_parameter(1,2)+Flow_rate_at; %(kg/s)

elseif part ==3
    Flow_rate_air=Process_parameter(1,3);
end




Section_bed=pi*0.25*Diameter_bed^2; %(m2)
Velocity_air=Flow_rate_air/(Section_bed*rho_air); %(m/s)

Conductivity_air=0.0262; %(W/m*K)
Viscosity_air= 1.94*1e-5; %(Pa*s)


Reynolds_particle=rho_air*Velocity_air*Diameter_eq/Viscosity_air; %(/)
Prandtl_etage= cp_a*Viscosity_air/Conductivity_air; %(/)

Nusselt_particle= 2 + Prandtl_etage^(2/5) *(0.43*Reynolds_particle^0.5 + 0.06*Reynolds_particle^(2/3)); %(/)

alpha= Nusselt_particle*Conductivity_air/Diameter_eq;

end

