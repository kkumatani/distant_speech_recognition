
function [y,val] = mynull( A, num, tol )
  
[U,W,V] = svd(A);
[rowN,colN] = size(A);
if (nargin == 1)
    num = 0;
end
if (  num > 0 )
  sIdx = colN-num+1;
  val = zeros(num,1);
else
  if rowN > 1, s = diag(W);
  elseif rowN == 1, s = W(1);
  else s = 0;
  end
  told = max(rowN,colN) * max(s) * eps( 'double' );
  if ( nargin < 3 || tol == 'd' )
    tol  = told;
  elseif( tol == 'f' )
    tol  = max(rowN,colN) * max(s) * eps( 'single' );
  elseif( tol == 's' )
    tol  = max(rowN,colN) * max(s) * eps( 'single' ) / 8;  
  end

  fprintf('Threshold for nullspace %e (%e)\n', tol, told );
  r = sum(s > tol);
  sIdx = r + 1;
  val = zeros(colN-sIdx+1,1);
end

y = V(:,sIdx:colN);
for i=sIdx:min(rowN,colN)
  val(i,1) = W(i,i);
end
