img = imread('C:\Users\jhoni\OneDrive\Documents\MATLAB\truchaR-removebg-preview.png');
%disp(img)
% Mostrar la imagen original
imshow(img);
title('Imagen original');
%Tamaño de la imagen
disp("tamaño de imagen")
whos("img")
% Definir el filtro
filtro = [-2 -1 0;
          -1 1 1;
          0 1 2];

% Normalizar el filtro
filtro = filtro / sum(filtro(:));

% Aplicar la convolución a la imagen y el filtro
img_filt = convn(img, filtro, 'same');
img_filt = uint8(img_filt);


%convertir la matriz en un vector
disp("vector de omg")
v1 =reshape(double(img_filt), 1, []);
for i = 1:length(v1)
  if v1(i) == 1
    v1(i) = 0.99;
  elseif v1(i) == 0
    v1(i) = 0.1;
  end
end
 
%fprintf('Vector x1: \n')
%fprintf('%f ', v1)
%fprintf('\n')

% Mostrar la imagen filtrada
subplot(1,2,1);
imshow(img);
title('Imagen Original');
subplot(1,2,2);
imshow(img_filt);
title('Imagen Filtrada');
disp("tamaño")
whos("img_filt")

%--------------------------------------------------------------------------
%x1=[0.99 0.1 0.1 0.1 0.99 0.1 0.99 0.99 0.99 0.1 0.1 0.99 0.99 0.99 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.99 0.99 0.99 0.1 0.1 0.99 0.99 0.99 0.1 0.1 0.99 0.99 0.99 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.99 0.99 0.99 0.99 0.1 0.99 0.99 0.99 0.99 0.1 0.1 0.1 0.99 0.99 0.1 0.99 0.99 0.99 0.99 0.1 0.99 0.99 0.99 0.99 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.99 0.99 0.1 0.99 0.99 0.99 0.99 0.1 0.99 0.99 0.99 0.99 0.1 0.99 0.99 0.99 0.99 0.1 0.99 0.99 0.99 0.99 0.1 0.99 0.99 0.1 0.1 0.1 0.1 0.1 0.99 0.1 0.1 0.1 0.99 0.1 0.99 0.99 0.99 0.1 0.1 0.99 0.99 0.99 0.1 0.1 0.99 0.99 0.99 0.1 0.1 0.99 0.99 0.99 0.1 0.1 0.99 0.99 0.99 0.1 0.99 0.1 0.1 0.1 0.99 0.1 0.99 0.99 0.99 0.1 0.1 0.99 0.99 0.99 0.1 0.1 0.99 0.99 0.99 0.1 0.1 0.99 0.99 0.99 0.1 0.1 0.99 0.99 0.99 0.1 0.1 0.99 0.99 0.99 0.1 0.99 0.1 0.1 0.1 0.99];
yd =[0.1 0.1 0.99];

num_neuronas_capa_oculta = 5;

whj = zeros(num_neuronas_capa_oculta, length(x1));
for i = 1:num_neuronas_capa_oculta
    whj(i,:) = -1 + (1+1)*rand(1,length(x1));
end
tam1= length(whj(1,:));
lon_whj = tam1*num_neuronas_capa_oculta; %cantidad de pesos

alfa =0.3;
casos =0;
ep =1;
beta =0.3;
errores =[];
num_epocas =1;

th = rand(1,num_neuronas_capa_oculta); %umbrales de cada neurona en la capa oculta
wo = rand(1,length(yd)*num_neuronas_capa_oculta);
tk = rand(1,length(yd));

nethj = zeros([1,num_neuronas_capa_oculta]);
yhj = zeros([1,num_neuronas_capa_oculta]);
dhj = zeros([num_neuronas_capa_oculta,length(x1)]);
netok =zeros([1,length(yd)]); %vector de sumatorias netok
yok = zeros([1,length(yd)]); % vector para la salida yok
dok = zeros([1,length(yd)]); % vector para los errores delta


while ep >=0.0000001
    for nt = 1:1:length(nethj)
        for j = 1:1:length(whj(1,:))
            nethj(nt) = nethj(nt) + whj(nt,j)*x1(j)+th(nt);
        end
    end

    for i=1:1:num_neuronas_capa_oculta
        yhj(i)=1/(1+exp(-nethj(i)));
    end
    
    %----------------------------------------------------------------------
    %calculos neuronas de salida.
    for k =1:1:length(wo)
        for j =1:1:length(yd)
            for yh=1:1:num_neuronas_capa_oculta
                 netok(1,j)=wo(1,k)*yhj(1,yh)+tk(j);
                 yok(1,j)=1/(1+exp(-netok(1,j)));
                 dok(1,j)=(yd(1,j)-yok(1,j))*yok(1,j)*(1-yok(1,j));  % Errores delta 
            end
        end
    end
    %----------------------------------------------------------------------
  
    %Propagación del error hacia atrás.
    for i = 1:length(x1)
         for j = 1:length(yd)*num_neuronas_capa_oculta
            dk = mod(j-1,length(yd))+1;
            yh = ceil(j/length(yd));

            dhj(yh,i) = dhj(yh,i) + x1(i)*(1-x1(i))*dok(dk)*wo(j);

         end
    end
    %----------------------------------------------------------------------
    %Actualizacion de pesos de salida.
     for k =1:1:length(wo)
        for j =1:1:length(yd)
                for yh=1:1:num_neuronas_capa_oculta
                    wo(1,k)=wo(1,k)+alfa*dok(1,j)*yhj(1,yh); %actualizacion de pesos capa de salida
                end
                 tk(1,j)=tk(1,j)+alfa*dok(1,j)*1;              %actualizacion de pesos umbrales
        end
    end
    %----------------------------------------------------------------------
    %Actualizacion de pesos de entrada.
    for i = 1:1:num_neuronas_capa_oculta
        for j = 1:length(x1)
            whj(i,:) = whj(i,:) + alfa*dhj(i,j)*x1(j);
            for t =1:1:length(th)
                 
                 th(1,t)=th(1,t)+alfa*(dhj(i,j))/num_neuronas_capa_oculta;
            end
        end
    end
    
    for i=1:1:length(dok)
        ep=1/2*(dok(1,i)^2);
    end
    %agregar erro a la lista
    errores(num_epocas)=ep;
    num_epocas = num_epocas +1;

    casos = casos+1;
end


tolerancia = 0.05; % porcentaje de tolerancia

diferencia = abs(yok - yd);
porcentaje = 100 * diferencia / max(abs(yd), abs(yok));

if diferencia <= tolerancia * max(abs(yd), abs(yok))
    disp('Los numeros son aproximadamente iguales.');
    fprintf('Los numeros son %.2f%% iguales.\n', 100 - porcentaje);
else
    disp('Los numeros no son aproximadamente iguales.');
    fprintf('Los numeros son %.2f%% diferentes.\n', porcentaje);
end

disp("---------------------------------------------------------------------");
disp("NUEVOS PESOS DE ENTRADA");
%whjfinal = reshape(whj ,[num_neuronas_capa_oculta,length(whj)]);
%disp(whj)
% Abrir el archivo en modo de escritura
archivo = fopen('datos2.txt', 'w');

% Escribir los datos en el archivo
for i = 1:size(whj, 1)
    fprintf(archivo, '%f ', whj(i, :));
    fprintf(archivo, '\n');
end

% Cerrar el archivo
fclose(archivo);

disp("PARA UMBRALES TH")
disp(th)

disp("NUEVOS PESOS DE SALIDA WO")
disp(wo)
disp("NUEVOS PESOS DE UMBRALES TK")
disp(tk)

disp("SALIDA OBTENIDA yok")
disp(yok)
disp("epocas")
disp(casos)

% Graficar épocas versus error
figure
plot(1:length(errores), errores)
title('Épocas versus error')
xlabel('Épocas')
ylabel('Error')

plot(yd, 'o-', 'LineWidth', 2, 'MarkerSize', 10)
hold on
plot(yok, 's-', 'LineWidth', 2, 'MarkerSize', 10)
legend('yd', 'yok')  % Añadir leyenda
title('Valores de yd y yok')
xlabel('Índice')
ylabel('Valor')



