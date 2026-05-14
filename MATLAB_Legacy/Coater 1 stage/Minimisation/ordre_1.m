
function somme_ecart=ordre_1(k_spray)


global Parameter MassCoating_exp
global k

Run=k;



DMC=Parameter.EC_conc(Run)*0.01;
Time_EndSpraying=Parameter.Qty_solution(Run)*1000*60/Parameter.Spray_rate(Run);
Batch_size=Parameter.Batch_size(Run);

Yp_exp=MassCoating_exp.Coating_Mass(Run);

Parameter.Spray_rate(Run)=Parameter.Spray_rate(Run)*1000/60;
somme_ecart=((DMC*Parameter.Spray_rate(Run)/k_spray)*(1-exp(-Time_EndSpraying*k_spray/Batch_size)) -Yp_exp )^2;




end

