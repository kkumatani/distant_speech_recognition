% create prototypes 
% Filter Bank Design for Subband Adaptive Beamforming
function [h,g] = NyquistFB( fftLen, m, r, wpW, disp )
% 
% last changed: $Date$
% revision: $Rev$
% 
% 

if( m < 2 )
  fprintf(1, 'm = %d < 2?', m);
end

x_m   = m;                     % factor of a filter length
D     = ( fftLen ) / 2 ^ r;    % decimation factor
L_h   = fftLen*m;              % length of analysis filter
L_g   = L_h;                   % length of synthesis filter
if x_m == 1
  md = 0;
else
  md = L_h/2;                  % group delay of analysis fb
end
tau_h = floor(L_h/2);          % group delay of analysis fb
tau_g = floor(L_g/2);          % group delay of synthesis fb
tau_t = md + tau_g;            % total filterbank delay
w_p   = pi/( wpW*fftLen);      % passband cut-off frequency
w = 1.0;

%-- print  command line arguments
fprintf(1, 'D     = %d\n', D);
fprintf(1, 'fftLen= %d\n', fftLen);
fprintf(1, 'L_h = L_g  = %d\n', L_h, L_g);
fprintf(1, 'md = %d\n', md);
fprintf(1, 'tau_g = %d\n', tau_g);
fprintf(1, '\n');

% analysis filter prototype ----------------------------------------------------
fprintf(1, 'calculating analysis prototype...');

A = zeros(L_h, L_h);                    % A is (L_h x L_h) hermitian matrix
b = zeros(L_h, 1);
C = zeros(L_h, L_h);

for m = 1:L_h
  for n = 1:L_h
    i = m - 1;
    j = n - 1;

    factor = -1;
    if (mod((j-i),D) == 0)
      factor = D-1;
    end

    if ( (j-i) == 0)
       C(m,n) = factor / D;
    else
       C(m,n) = factor * ( sin( pi * (j-i) / D ) ) / ( pi * (j-i) );
    end
    
    if ( (j-i) == 0)
      A(m,n) = 1;
    else
      A(m,n) = sin( w_p * (j-i) ) / ( w_p * (j-i) );
    end
  end

  if (tau_h-i == 0)
    b(m) = 1;
  else
    b(m) = sin( w_p * (tau_h-i) ) / ( w_p * (tau_h-i) );
  end

%  fprintf(1, '.');
end

% delete the rows and columns of C corresponding to the components of h = 0
delC = zeros( L_h - x_m + 1, L_h - x_m + 1 );
delA = zeros( L_h - x_m + 1, L_h - x_m + 1 );
delb = zeros( L_h - x_m + 1, 1 );
m1 = 1;
for m = 1:L_h
  if ( (m-1) == md || mod( m-1, fftLen )~=0 )
    n1 = 1;
    for n = 1:L_h
      if (  (n-1) == md || mod( n-1, fftLen )~=0 )
        delA(m1,n1) = A(m,n);
        delC(m1,n1) = C(m,n);
        n1 = n1 + 1;
      end
    end
    delb(m1) = b(m);
    m1 = m1 + 1;
  end
end

if (  rank(delC) == size(delC,1) )
    % take an eigen vector corresponding to the smallest eigen value.
    [eVec,eVal] = eig(delC);
    % take eigen vectors as basis
    fprintf('\nmin eigen val %e\n',eVal(1,1));
    rh = eVec(:,1); % eigen values are sorted in the ascending order.
else
    nulldelC = mynull( delC );
    if ( size(nulldelC,2) == 0 )
        fprintf('the nullity of C is 0 (%d %d %d)\n', fftLen, x_m, r);
        fprintf('try another method\n');
        return
    end
    fprintf( 'the nullity of C %d (%d %d %d)\n', size(nulldelC,2), fftLen, ...
         x_m, r );
    % In general, null(delP) is not a square matrix.
    % We don't want to use a peseude inversion matrix as much as possible.
    T1    = delA * nulldelC;
    T1_2  = nulldelC' * T1;
    rankN = rank( T1_2 );
    if ( rankN == size( T1_2, 1 ) )
      x = T1_2 \ ( nulldelC' * delb );
    else
      fprintf( 'use pseudo-inverse matrix because %d < %d\n', rankN, size( T1_2, 1 ) );
      x = pinv( T1 ) * delb;
    end
    rh = nulldelC * x;
    clear nulldelC;
end

% re-assemble the complete prototype
h = zeros(L_h, 1);
k = 1;
for m = 1:L_h
  if( (m-1) ~=  md && mod( m-1, fftLen )==0 )
    h(m) = 0;
  else
    h(m) = rh(k);
    k = k + 1;
  end
end

alpha = h' * A * h - 2 * h' * b + 1; % 10 * log10( h' * A * h - 2 * h' * b + 1 );
beta  = h' * C * h;
clear A;
clear b;
clear C;
clear delA;
clear delC;
clear delb;

% synthesis filter prototype ---------------------------------------------------
fprintf('\ncalculating synthesis prototype...');
E = zeros(L_g, L_g);
f = zeros(L_g, 1);
P = zeros(L_g, L_g);

for m = 1:L_g
  for n = 1:L_g
    m1 = m - 1;
    n1 = n - 1;

    for k = 1:(2*L_g/fftLen+1)
      if (k*fftLen-m < 1   ) continue; end
      if (k*fftLen-n < 1   ) continue; end
      if (k*fftLen-m > L_h ) continue; end
      if (k*fftLen-n > L_h ) continue; end
      
      E(m,n) = E(m,n) + h(k*fftLen-m) * h(k*fftLen-n);
    end

    factor = -1;
    if (mod(m1-n1, D) == 0)
      factor = D-1;
    end

    for k = -max(L_g, L_h):max(L_g, L_h)
      if (k+n < 1) continue; end
      if (k+m < 1) continue; end
      if (k+n > L_h) continue; end
      if (k+m > L_h) continue; end
      
      P(m,n) = P(m,n) +h(k+n) * h(k+m) * factor;
    end
  end

  if (tau_t-m < 1) continue; end
  if (tau_t-m > L_h) continue; end
  f(m) = h(tau_t-m);
end

E = ( ( fftLen / D ) * ( fftLen / D ) ) .*E;
f = ( fftLen / ( pi * D ) ).*f;
P = ( fftLen / ( D * D ) ).*P;

% shift a time-reversed version of h and make a matrix.
% The k-th row of the matrix indicates h_k
rowN = 2 * x_m - 1;
H  = zeros( rowN, L_g ); % a row vector corresponds to h_k in the report
sIdx = fftLen; % + 1;
eIdx = sIdx - L_g + 1; % + 1;
for m = 1:rowN
  s = sIdx;
  if ( s < 1 )
    s = 1;
  end
  if ( s > L_g )
    s = L_g;
  end
  e = eIdx;
  if ( e < 1 )
    e = 1;
  end
  if ( e > L_g )
    e = L_g;
  end
  H(m,e:s) = h(s:-1:e)';
  sIdx = sIdx + fftLen;
  eIdx = eIdx + fftLen;
end
 
C0 = zeros( rowN, 1);
C0(x_m) =  D * 1.0 / fftLen; % C0(x_m) = h(md+1);

sizeP = size(P,1);
rank_P = rank(P);
if ( rank_P == sizeP )
  fprintf('with Lagrange multiplier...\n');
  invP = inv( P );
  g  = invP * H' * inv( H * invP * H' ) * C0;
elseif ( rank_P <= ( sizeP - rowN ) )
  fprintf('with the null space...\n');
  nullP = mynull( P );
  fprintf( 'the nullity of P %d (%d %d %d)\n', size(nullP,2), fftLen, ...
         x_m, r );
  y = pinv( H * nullP ) * C0;
  g = nullP * y;
else
  % will not find enough bases of the null space to achieve the
  % Nyquist(M) filter bank.
  fprintf('with SVD (rank(P)=%d)...\n',rank_P);
  [UP,WP,VP] = svd( P );
  fprintf( 'sigular values ');
  for i=(sizeP-rowN+1):sizeP
    fprintf( '%e ', WP(i,i));
  end
  fprintf( '\n');
  pnullP = VP(:,(sizeP-rowN+1):sizeP);
  y = ( H * pnullP )\ C0;
  g = pnullP * y;
end

gamma = g' * E * g - 2 * g' * f + 1;
epsir = g' * P * g;

fprintf('\n');
fprintf('b=%e e=%e\n', 10*log10(beta), 10*log10(epsir));
fprintf('NyquistFB M=%d m=%d r=%d\n', fftLen, x_m, r);

%-- save prototypes to disk ----------------------------------------------------
outdirname = sprintf('prototype.ny');
if (~exist(outdirname, 'dir'))
  mkdir(outdirname);
end

filename = sprintf('%s/M=%d-m=%d-r=%d.m',outdirname, fftLen, x_m, r);
fid = fopen(filename, 'w');
if (fid <= 0)
  error( 'could not open file %s', filename );
end
fprintf(fid, '%e ', h);
fprintf(fid, '\n');
fprintf(fid, '%e ', g);
fclose(fid);
fprintf('saved %s...\n',filename);

% plot results...
if( disp ==0 )
  return
end
figdirname = sprintf('fig.ny');
if (~exist(figdirname, 'dir'))
  mkdir(figdirname);
end
h_x_g = convn(h,g);

figh = plot(h);
filename = sprintf('%s/IRA_M=%d-m=%d-r=%d.pdf',figdirname, fftLen, x_m, r);
saveas(figh,filename);

figh =figure;
plot( g );
filename = sprintf('%s/IRS_M=%d-m=%d-r=%d.pdf',figdirname, fftLen, x_m, ...
                   r);
saveas(figh,filename);

figh =figure;
plot( h_x_g );
filename = sprintf('%s/IRE_M=%d-m=%d-r=%d.pdf',figdirname, fftLen, x_m, ...
                   r);
saveas(figh,filename);

figh =figure;
freqz( h );
filename = sprintf('%s/FRA_M=%d-m=%d-r=%d.pdf',figdirname, fftLen, x_m, ...
                   r);
saveas(figh,filename);

figh =figure;
freqz( g );
filename = sprintf('%s/FRS_M=%d-m=%d-r=%d.pdf',figdirname, fftLen, x_m, ...
                   r);
saveas(figh,filename);

figh =figure;
freqz( h_x_g );
filename = sprintf('%s/FRE_M=%d-m=%d-r=%d.pdf',figdirname, fftLen, x_m, ...
                   r);
saveas(figh,filename);
