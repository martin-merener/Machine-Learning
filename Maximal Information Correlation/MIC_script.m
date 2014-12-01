% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%

% Maximal Information Coefficient
% From: Detecting Novel Associations in Large Datasets
%   at: http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3325791/

N = 2000;

X = 10*rand(N,1)-5;
noise = (2*rand(N,1)-1)/2;
Y = MICfunTest(X) + noise;

scatter(X,Y);

R = 40;
M = zeros(R-1);

MIC = 0;
for n_x = 2:R
    for n_y = 2:R
        P = distribution(X,Y,n_x,n_y);
        MI = MICmutualInformation(P);
        m_xy = MI/log(min(n_x,n_y));
        M(n_x-1,n_y-1) = m_xy;
        if m_xy>MIC
            n_x_opt = n_x;
            n_y_opt = n_y;
            MIC = m_xy;
        end
    end
end

P = distribution(X,Y,n_x_opt,n_y_opt);

figure
subplot(2,2,1);
scatter(X,Y);
title('scatter plot')

subplot(2,2,2);
imagesc(P);
title('distribution on (x,y)-grid maximizing m_{x,y}')

subplot(2,2,3);
imagesc(M);
title('M =  m_{x,y}, for x=2,...,40, y=2,...,40')

text(50,3,strcat('MIC =',num2str(MIC)))
text(50,10,strcat('Best grid x-by-y:',num2str(n_x_opt),'-by-',num2str(n_y_opt)));

colormap(gray);



disp(strcat('MIC =',num2str(MIC)));
disp(strcat('Best grid x-by-y:',num2str(n_x_opt),'-by-',num2str(n_y_opt)));


