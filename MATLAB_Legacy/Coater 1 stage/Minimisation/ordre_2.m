
function somme_ecart=ordre_2(k_spray)


global Parameter MassCoating_exp
global k

Run=k;



DMC=Parameter.EC_conc(Run)*0.01;
Time_EndSpraying=Parameter.Qty_solution(Run)*1000*60/Parameter.Spray_rate(Run);
Batch_size=Parameter.Batch_size(Run);

Yp_exp=MassCoating_exp.Coating_Mass(Run);


somme_ecart=( (DMC*Parameter.Spray_rate(Run)/k_spray)^0.5 *tan...
    (((k_spray*DMC*Parameter.Spray_rate(Run))^0.5)*Time_EndSpraying/Batch_size ) -Yp_exp )^2;




end

