function R_D=Calcul_DryingRate(Moisture_particle,Gas_moisture,Temp_particle,Temp_air,part)





global Surface_particle Mass_particle Diameter_eq  ...
     Diameter_bed

global Gas_constant  Molecular_weight_acetone ...
     Molecular_weight_air

global  Pressure

    


alpha_langmuir=0.05;

Section_bed=pi*0.25*Diameter_bed^2; %(m2)
%Surface_particle=0.5*pi*Diameter_particle^2 + pi*Diameter_particle*Length_particle; %(m2) %cylindre
Surface_particle= pi*Diameter_eq^2;

Pressure_vap_particle= 10^(7.1327 - 1219.97/(230.653+Temp_particle-273.15)); %(mmHg)
Pressure_vap_particle=Pressure_vap_particle*133.322; %(Pa)

Pressure_vap_gas= Pressure*Gas_moisture / ((Molecular_weight_acetone/Molecular_weight_air) + Gas_moisture); %(Pa)

alpha=Calcul_MassTransfer(part);


R_D=  Moisture_particle/(abs(Moisture_particle)+alpha_langmuir) * alpha * Surface_particle...
    * (Pressure_vap_particle-Pressure_vap_gas)  / (Mass_particle*(Gas_constant/Molecular_weight_acetone)...
    * 0.5*(Temp_particle+Temp_air));
end
