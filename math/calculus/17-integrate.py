#!/usr/env/bin python3
def poly_integral(poly, C=0):
    if (not isinstance(poly, list) or
       not all(isinstance(coef, (int, float)) for coef in poly) or
       len(poly) == 0 or not isinstance(C, (int, float))):
        return None
    
    new_list =[poly[i] / (i + 1)  for i in range(len(poly))]

    return [C] + [int(x) if x.is_integer() else x for x in new_list]
