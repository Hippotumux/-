function [p] = Wb_2_Param(W,b,n)
   L = length(n);
   idx = 1;
   for l = 2:L
      p(idx:idx+n(l)*n(l-1)-1) = W{l}(:); idx = idx+n(l)*n(l-1);
      p(idx:idx+n(l)-1)        = b{l}(:); idx = idx+n(l);
   end
end