//Copyright 2024 Andrey S. Ionisyan (anserion@gmail.com)
//
//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.

unit Unit1;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Forms, Controls, Graphics, Dialogs, StdCtrls, ExtCtrls,
  ExtDlgs, FPCanvas, LCLintf, LCLType;

type

  { TForm1 }

  TForm1 = class(TForm)
    Bevel1: TBevel;
    Bevel_receptors: TBevel;
    BTN_out_gen_to_receptors: TButton;
    BTN_stop_train_generator: TButton;
    BTN_s_clear1: TButton;
    BTN_s_clear2: TButton;
    BTN_s_noise: TButton;
    BTN_train_discriminator: TButton;
    BTN_forward_discriminator: TButton;
    BTN_train_generator: TButton;
    BTN_train_art: TButton;
    BTN_nw_reset_discriminator: TButton;
    BTN_nw_reset_generator: TButton;
    BTN_shake_generator: TButton;
    BTN_stop_train_art: TButton;
    BTN_s_clear0: TButton;
    BTN_samples_load: TButton;
    BTN_nw_save: TButton;
    BTN_nw_load: TButton;
    BTN_forward_all: TButton;
    BTN_s_save: TButton;
    BTN_s_random: TButton;
    BTN_s_load: TButton;
    BTN_stop_train_discriminator: TButton;
    BTN_shake_discriminator: TButton;
    BTN_out_gen_save: TButton;
    CB_contrast: TCheckBox;
    CB_autolevel: TCheckBox;
    CB_samples_L1_L7: TCheckBox;
    CB_seed_from_S: TCheckBox;
    Edit_cur_sample: TEdit;
    Edit_n_cycles_Discriminator: TEdit;
    Edit_contrast: TEdit;
    Edit_target_Discriminator: TEdit;
    Edit_n_cycles_generator: TEdit;
    Edit_n_cycles_art: TEdit;
    Edit_N_L4: TEdit;
    Edit_N_L5: TEdit;
    Edit_N_L6: TEdit;
    Edit_L7_out: TEdit;
    Edit_N_L1: TEdit;
    Edit_N_L2: TEdit;
    Edit_N_L3: TEdit;
    Edit_target_art: TEdit;
    Edit_target_Discriminator_delta: TEdit;
    Label1: TLabel;
    Label10: TLabel;
    Label11: TLabel;
    Label12: TLabel;
    Label13: TLabel;
    Label14: TLabel;
    Label15: TLabel;
    Label16: TLabel;
    Label17: TLabel;
    Label18: TLabel;
    Label3: TLabel;
    Label4: TLabel;
    Label5: TLabel;
    Label6: TLabel;
    Label7: TLabel;
    Label8: TLabel;
    Label9: TLabel;
    Label_Layer3: TLabel;
    Label2: TLabel;
    OpenPictureDialog: TOpenPictureDialog;
    PB_generator: TPaintBox;
    PB_receptors: TPaintBox;
    SB_samples: TScrollBar;
    Timer_generator: TTimer;
    Timer_art: TTimer;
    Timer_discriminator: TTimer;
    procedure BTN_out_gen_to_receptorsClick(Sender: TObject);
    procedure BTN_samples_loadClick(Sender: TObject);
    procedure BTN_forward_discriminatorClick(Sender: TObject);
    procedure BTN_forward_allClick(Sender: TObject);
    procedure BTN_out_gen_saveClick(Sender: TObject);
    procedure BTN_stop_train_generatorClick(Sender: TObject);
    procedure BTN_s_clear1Click(Sender: TObject);
    procedure BTN_s_clear2Click(Sender: TObject);
    procedure BTN_s_noiseClick(Sender: TObject);
    procedure BTN_train_discriminatorClick(Sender: TObject);
    procedure BTN_train_artClick(Sender: TObject);
    procedure BTN_nw_loadClick(Sender: TObject);
    procedure BTN_nw_reset_discriminatorClick(Sender: TObject);
    procedure BTN_nw_reset_generatorClick(Sender: TObject);
    procedure BTN_nw_saveClick(Sender: TObject);
    procedure BTN_shake_generatorClick(Sender: TObject);
    procedure BTN_shake_discriminatorClick(Sender: TObject);
    procedure BTN_stop_train_artClick(Sender: TObject);
    procedure BTN_stop_train_discriminatorClick(Sender: TObject);
    procedure BTN_s_clear0Click(Sender: TObject);
    procedure BTN_s_loadClick(Sender: TObject);
    procedure BTN_s_randomClick(Sender: TObject);
    procedure BTN_s_saveClick(Sender: TObject);
    procedure BTN_train_generatorClick(Sender: TObject);
    procedure CB_autolevelChange(Sender: TObject);
    procedure CB_contrastChange(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure PB_generatorPaint(Sender: TObject);
    procedure PB_receptorsPaint(Sender: TObject);
    procedure SB_samplesChange(Sender: TObject);
    procedure Timer_artTimer(Sender: TObject);
    procedure Timer_discriminatorTimer(Sender: TObject);
    procedure Timer_generatorTimer(Sender: TObject);
  private

  public

  end;

const
  s_width=128;//64;
  s_height=128;//64;
  alpha_BPA_generator=0.01;
  alpha_BPA_discriminator=0.01;
  alpha_BPA_art=0.01;
  alpha_shake_generator=0.01;
  alpha_shake_discriminator=0.01;

type
  TIntegerVector = array of integer;
  TRealVector = array of real;
  TRealMatrix = array of TRealVector;

var
  Form1: TForm1;
  receptorsBitmap,generatorBitmap: TBitmap;
  img_buffer:array[0..2047,0..2047]of real;
  S_elements,Target_elements,Z_elements,tmp_elements:TRealMatrix;
  samples:array of TRealMatrix;
  shuffle:array of integer;

  n_train_cycles_discriminator,train_iteration_discriminator:integer;
  n_train_cycles_art,train_iteration_art:integer;
  n_train_cycles_generator,train_iteration_generator:integer;
  target_discriminator,target_discriminator_delta,target_art:real;

  n_generator_inputs:integer;
  generator_input:TRealVector;

  n_L1,n_L2,n_L3:integer;

  L1_w:TRealMatrix;
  L1_out:TRealVector;

  L2_w:TRealMatrix;
  L2_out:TRealVector;

  L3_w:TRealMatrix;
  L3_out:TRealVector;
  L3_target:TRealVector;

  sigma1:TRealVector;
  sigma2:TRealVector;
  sigma3:TRealVector;

  n_L4,n_L5,n_L6,n_L7:integer;

  L4_w:TRealMatrix;
  L4_out:TRealVector;

  L5_w:TRealMatrix;
  L5_out:TRealVector;

  L6_w:TRealMatrix;
  L6_out:TRealVector;

  L7_w:TRealMatrix;
  L7_out:TRealVector;
  L7_target:TRealVector;

  sigma4:TRealVector;
  sigma5:TRealVector;
  sigma6:TRealVector;
  sigma7:TRealVector;

implementation

{$R *.lfm}

function sigmoid(x:real):real;
begin sigmoid:=1/(1+exp(-x)); end;

function der_sigmoid(y:real):real;
begin der_sigmoid:=y*(1-y); end;

function tanh(x:real):real;
begin tanh:=(exp(x)-exp(-x))/(exp(x)+exp(-x)); end;

function der_tanh(y:real):real;
begin der_tanh:=1-y*y; end;

function ReLU(x:real):real;
begin if x<0 then ReLU:=0.01*x else ReLU:=x; end;

function der_ReLU(y:real):real;
begin if y<=0 then der_ReLU:=0.01 else der_ReLU:=1; end;

function activation(x:real):real;
begin activation:=tanh(x); end;

function der_activation(y:real):real;
begin der_activation:=der_tanh(y); end;

procedure nw_generator_allocation;
var k:integer;
begin
  SetLength(L1_w,n_L1);
  for k:=0 to n_L1-1 do SetLength(L1_w[k],n_generator_inputs);
  SetLength(L1_out,n_L1);
  SetLength(sigma1,n_L1);

  SetLength(L2_w,n_L2);
  for k:=0 to n_L2-1 do SetLength(L2_w[k],n_L1);
  SetLength(L2_out,n_L2);
  SetLength(sigma2,n_L2);

  SetLength(L3_w,n_L3);
  for k:=0 to n_L3-1 do SetLength(L3_w[k],n_L2);
  SetLength(L3_out,n_L3);
  SetLength(sigma3,n_L3);
  SetLength(L3_target,n_L3);
end;

procedure nw_discriminator_allocation;
var k:integer;
begin
  SetLength(L4_w,n_L4);
  for k:=0 to n_L4-1 do SetLength(L4_w[k],n_generator_inputs);
  SetLength(L4_out,n_L4);
  SetLength(sigma4,n_L4);

  SetLength(L5_w,n_L5);
  for k:=0 to n_L5-1 do SetLength(L5_w[k],n_L4);
  SetLength(L5_out,n_L5);
  SetLength(sigma5,n_L5);

  SetLength(L6_w,n_L6);
  for k:=0 to n_L6-1 do SetLength(L6_w[k],n_L5);
  SetLength(L6_out,n_L6);
  SetLength(sigma6,n_L6);

  SetLength(L7_w,n_L7);
  for k:=0 to n_L7-1 do SetLength(L7_w[k],n_L6);
  SetLength(L7_out,n_L7);
  SetLength(sigma7,n_L7);
  SetLength(L7_target,n_L7);
end;

procedure forward_calc(x,y:TRealVector; w:TRealMatrix);
var i,k,n_neurons,n_inputs:integer; scalar:real;
begin
  n_inputs:=length(x); n_neurons:=length(y);
  for k:=0 to n_neurons-1 do
  begin
    scalar:=0;
    for i:=0 to n_inputs-1 do scalar:=scalar+w[k,i]*x[i];
    y[k]:=activation(scalar);
  end;
end;

procedure BackTraceError_out_calc(y,target,sigma:TRealVector);
var k,n_neurons:integer; error_target_to_out:real;
begin
  n_neurons:=length(y);
  for k:=0 to n_neurons-1 do
  begin
    error_target_to_out:=-(target[k]-y[k]);
    sigma[k]:=error_target_to_out*der_activation(y[k]);
  end;
end;

procedure BackTraceError_middle_calc(y_curr:TRealVector;
                                     w_next:TRealMatrix;
                                     sigma_next,sigma_curr:TRealVector);
var i,k,n_neurons_next,n_neurons_curr:integer; error_next_to_curr:real;
begin
  n_neurons_curr:=length(sigma_curr); n_neurons_next:=length(sigma_next);
  for i:=0 to n_neurons_curr-1 do
  begin
    error_next_to_curr:=0;
    for k:=0 to n_neurons_next-1 do
      error_next_to_curr:=error_next_to_curr+sigma_next[k]*w_next[k,i];
    sigma_curr[i]:=error_next_to_curr*der_activation(y_curr[i]);
  end;
end;

procedure BackTracetrain_calc(x,sigma:TRealVector; w:TRealMatrix; alpha:real);
var i,k,n_neurons,n_inputs:integer;
begin
  n_inputs:=length(x); n_neurons:=length(sigma);
  for i:=0 to n_neurons-1 do
    for k:=0 to n_inputs-1 do
      w[i,k]:=w[i,k]-alpha*sigma[i]*x[k];
end;

procedure shuffleIntegerVector(vector:TIntegerVector);
var n,k,k1,k2:integer; tmp:integer;
begin
  n:=length(vector);
  //for k:=0 to n-1 do vector[k]:=k mod n;
  for k:=1 to n do
  begin
    k1:=random(n); k2:=random(n);
    tmp:=vector[k1]; vector[k1]:=vector[k2]; vector[k2]:=tmp;
  end;
end;

procedure valueToVector(vector:TRealVector; value:real);
var i,n:integer;
begin
  n:=length(vector);
  for i:=0 to n-1 do vector[i]:=value;
end;

procedure valueToMatrix(matrix:TRealMatrix; value:real);
var i,n:integer;
begin
  n:=length(matrix);
  for i:=0 to n-1 do valueToVector(matrix[i],value);
end;

procedure randomToVector(vector:TRealVector; min_value,max_value:real);
var i,n:integer;
begin
  n:=length(vector);
  for i:=0 to n-1 do vector[i]:=min_value+(max_value-min_value)*random;
end;

procedure randomToMatrix(matrix:TRealMatrix; min_value,max_value:real);
var i,n:integer;
begin
  n:=length(matrix);
  for i:=0 to n-1 do randomToVector(matrix[i],min_value,max_value);
end;

procedure shakeVector(vector:TRealVector; value:real);
var i,n:integer;
begin
  n:=length(vector);
  for i:=0 to n-1 do vector[i]:=vector[i]+2*value*(random-0.5);
end;

procedure shakeMatrix(matrix:TRealMatrix; value:real);
var i,n:integer;
begin
  n:=length(matrix);
  for i:=0 to n-1 do shakeVector(matrix[i],value);
end;

procedure noiseToVector(vector:TRealVector; noise_value:real);
var i,n:integer;
begin
  n:=length(vector);
  for i:=0 to n-1 do
    if random<=noise_value then vector[i]:=2.0*random-1.0;
end;

procedure noiseToMatrix(matrix:TRealMatrix; noise_value:real);
var i,n:integer;
begin
  n:=length(matrix);
  for i:=0 to n-1 do noiseToVector(matrix[i],noise_value);
end;

procedure MatrixToVector(matrix:TRealMatrix; vector:TRealVector);
var i,j,n,m,cnt:integer;
begin
  n:=length(matrix); m:=length(matrix[0]); cnt:=0;
  for i:=0 to n-1 do
  for j:=0 to m-1 do
  begin
    vector[cnt]:=matrix[i,j];
    cnt:=cnt+1;
  end;
end;

procedure VectorToMatrix(vector:TRealVector; matrix:TRealMatrix);
var i,j,n,m,cnt:integer;
begin
  n:=length(matrix); m:=length(matrix[0]); cnt:=0;
  for i:=0 to n-1 do
  for j:=0 to m-1 do
  begin
    matrix[i,j]:=vector[cnt];
    cnt:=cnt+1;
  end;
end;

procedure VectorCpy(src,dst:TRealVector);
var i,n:integer;
begin
     n:=length(src);
     for i:=0 to n-1 do dst[i]:=src[i];
end;

procedure MatrixCpy(src,dst:TRealMatrix);
var i,n:integer;
begin
     n:=length(src);
     for i:=0 to n-1 do VectorCpy(src[i],dst[i]);
end;

function VectorMin(vector:TRealVector):real;
var i,n:integer; res:real;
begin
     n:=length(vector);
     res:=vector[0];
     for i:=0 to n-1 do
         if vector[i]<res then res:=vector[i];
     VectorMin:=res;
end;

function VectorMax(vector:TRealVector):real;
var i,n:integer; res:real;
begin
     n:=length(vector);
     res:=vector[0];
     for i:=0 to n-1 do
         if vector[i]>res then res:=vector[i];
     VectorMax:=res;
end;

function MatrixMin(matrix:TRealMatrix):real;
var i,n:integer; tmp,res:real;
begin
     n:=length(matrix);
     res:=matrix[0,0];
     for i:=0 to n-1 do
     begin
       tmp:=VectorMin(matrix[i]);
       if tmp<res then res:=tmp;
     end;
     MatrixMin:=res;
end;

function MatrixMax(matrix:TRealMatrix):real;
var i,n:integer; tmp,res:real;
begin
     n:=length(matrix);
     res:=matrix[0,0];
     for i:=0 to n-1 do
     begin
       tmp:=VectorMax(matrix[i]);
       if tmp>res then res:=tmp;
     end;
     MatrixMax:=res;
end;

procedure MatrixToBitmap(matrix:TRealMatrix; bitmap:TBitmap; contrast:real; autoflag:boolean);
var x,y,sx,sy:integer; dx,dy:real; C_min,C_max,deltaC,C:real;
    dst_bpp:integer; dst_ptr:PByte; R,G,B:byte;
    n,m:integer;
begin
  n:=length(matrix);
  m:=length(matrix[0]);
  dx:=bitmap.Width/n;
  dy:=bitmap.Height/m;

  C_min:=MatrixMin(matrix); C_max:=MatrixMax(matrix);
  deltaC:=C_max-C_min;
  if deltaC=0 then autoflag:=false;
  for x:=0 to n-1 do
  for y:=0 to m-1 do
  begin
    if autoflag
    then C:=((matrix[x,y]-C_min)/deltaC-0.5)*2.0
    else C:=matrix[x,y];

    //C:=(C-0.5)*contrast+0.5; //sigmoid activation
    C:=C*contrast; //tahh activation
    //if C<0 then C:=0; //sigmoid
    if C<-1 then C:=-1; //tanh
    if C>1 then C:=1;
    C:=(C+1.0)*0.5; //tanh
    for sx:=0 to trunc(dx) do
    for sy:=0 to trunc(dy) do
      img_buffer[trunc(sx+x*dx),trunc(sy+y*dy)]:=C;
  end;

  bitmap.BeginUpdate(false);
  dst_ptr:=bitmap.RawImage.Data;
  dst_bpp:=bitmap.RawImage.Description.BitsPerPixel div 8;
  for y:=0 to bitmap.height-1 do
  for x:=0 to bitmap.width-1 do
  begin
     R:=trunc(img_buffer[x,y]*255); G:=R; B:=R;
     dst_ptr^:=B; (dst_ptr+1)^:=G; (dst_ptr+2)^:=R; inc(dst_ptr,dst_bpp);
  end;
  bitmap.EndUpdate(false);
end;

procedure Forward_generator;
begin
  MatrixToVector(S_elements,generator_input);
  forward_calc(generator_input,L1_out,L1_w);
  forward_calc(L1_out,L2_out,L2_w);
  forward_calc(L2_out,L3_out,L3_w);
  VectorToMatrix(L3_out,Z_elements);
end;

procedure BackTraceError_generator;
begin
  MatrixToVector(Target_elements,L3_target);
  BackTraceError_out_calc(L3_out,L3_target,sigma3);
  BackTraceError_middle_calc(L2_out,L3_w,sigma3,sigma2);
  BackTraceError_middle_calc(L1_out,L2_w,sigma2,sigma1);
end;

procedure BackTraceTrain_generator;
begin
  MatrixToVector(S_elements,generator_input);
  BackTracetrain_calc(generator_input,sigma1,L1_w,alpha_BPA_generator);
  BackTracetrain_calc(L1_out,sigma2,L2_w,alpha_BPA_generator);
  BackTracetrain_calc(L2_out,sigma3,L3_w,alpha_BPA_generator);
end;

procedure Forward_Discriminator;
begin
  MatrixToVector(S_elements,generator_input);
  forward_calc(generator_input,L4_out,L4_w);
  forward_calc(L4_out,L5_out,L5_w);
  forward_calc(L5_out,L6_out,L6_w);
  forward_calc(L6_out,L7_out,L7_w);
end;

procedure BackTraceError_Discriminator;
begin
  L7_target[0]:=target_discriminator;
  BackTraceError_out_calc(L7_out,L7_target,sigma7);
  BackTraceError_middle_calc(L6_out,L7_w,sigma7,sigma6);
  BackTraceError_middle_calc(L5_out,L6_w,sigma6,sigma5);
  BackTraceError_middle_calc(L4_out,L5_w,sigma5,sigma4);
end;

procedure BackTraceTrain_Discriminator;
begin
  MatrixToVector(S_elements,generator_input);
  BackTracetrain_calc(generator_input,sigma4,L4_w,alpha_BPA_discriminator);
  BackTracetrain_calc(L4_out,sigma5,L5_w,alpha_BPA_discriminator);
  BackTracetrain_calc(L5_out,sigma6,L6_w,alpha_BPA_discriminator);
  BackTracetrain_calc(L6_out,sigma7,L7_w,alpha_BPA_discriminator);
end;

procedure Forward_all;
begin
  MatrixToVector(S_elements,generator_input);
  forward_calc(generator_input,L1_out,L1_w);
  forward_calc(L1_out,L2_out,L2_w);
  forward_calc(L2_out,L3_out,L3_w);
  forward_calc(L3_out,L4_out,L4_w);
  forward_calc(L4_out,L5_out,L5_w);
  forward_calc(L5_out,L6_out,L6_w);
  forward_calc(L6_out,L7_out,L7_w);
  VectorToMatrix(L3_out,Z_elements);
end;

procedure BackTraceError_all;
begin
  L7_target[0]:=target_art;
  BackTraceError_out_calc(L7_out,L7_target,sigma7);
  BackTraceError_middle_calc(L6_out,L7_w,sigma7,sigma6);
  BackTraceError_middle_calc(L5_out,L6_w,sigma6,sigma5);
  BackTraceError_middle_calc(L4_out,L5_w,sigma5,sigma4);
  BackTraceError_middle_calc(L3_out,L4_w,sigma4,sigma3);
  BackTraceError_middle_calc(L2_out,L3_w,sigma3,sigma2);
  BackTraceError_middle_calc(L1_out,L2_w,sigma2,sigma1);
end;

procedure BackTraceTrain_all;
begin
  MatrixToVector(S_elements,generator_input);
  BackTracetrain_calc(generator_input,sigma1,L1_w,alpha_BPA_art);
  BackTracetrain_calc(L1_out,sigma2,L2_w,alpha_BPA_art);
  BackTracetrain_calc(L2_out,sigma3,L3_w,alpha_BPA_art);

  BackTracetrain_calc(L3_out,sigma4,L4_w,alpha_BPA_art);
  BackTracetrain_calc(L4_out,sigma5,L5_w,alpha_BPA_art);
  BackTracetrain_calc(L5_out,sigma6,L6_w,alpha_BPA_art);
  BackTracetrain_calc(L6_out,sigma7,L7_w,alpha_BPA_art);
end;

{ TForm1 }

procedure TForm1.PB_receptorsPaint(Sender: TObject);
begin
  MatrixToBitmap(S_elements,receptorsBitmap,1,CB_autolevel.Checked);
  PB_receptors.Canvas.Draw(0,0,receptorsBitmap);
end;

procedure TForm1.SB_samplesChange(Sender: TObject);
begin
  MatrixCpy(samples[SB_samples.Position],S_elements);
  Edit_cur_sample.text:=IntToStr(SB_samples.Position+1)+'/'+IntToStr(length(samples));
  if CB_samples_L1_L7.Checked
  then BTN_forward_allClick(self)
  else PB_receptorsPaint(PB_receptors);
end;

procedure TForm1.PB_generatorPaint(Sender: TObject);
var contrast_value:real;
begin
  if CB_contrast.Checked
  then contrast_value:=StrToFloat(Edit_contrast.text)/100
  else contrast_value:=1;
  MatrixToBitmap(Z_elements,generatorBitmap,contrast_value,CB_autolevel.Checked);
  PB_generator.Canvas.Draw(0,0,generatorBitmap);
end;

procedure TForm1.FormCreate(Sender: TObject);
begin
  randomize;
  SetLength(S_elements,s_width,s_height);
  SetLength(Z_elements,s_width,s_height);
  SetLength(Target_elements,s_width,s_height);
  SetLength(tmp_elements,s_width,s_height);
  n_generator_inputs:=s_width*s_height;
  SetLength(generator_input,n_generator_inputs);
  SetLength(samples,0);
  SB_samples.max:=length(samples);
  receptorsBitmap:=TBitmap.Create;
  receptorsBitmap.SetSize(PB_receptors.width,PB_receptors.height);
  generatorBitmap:=TBitmap.Create;
  generatorBitmap.SetSize(PB_generator.width,PB_generator.height);
  Edit_target_art.text:=FloatToStr(0.75);
  Edit_target_Discriminator.text:=FloatToStr(0.5);
  Edit_target_Discriminator_delta.text:=FloatToStr(0.1);
  BTN_nw_reset_discriminatorClick(self);
  BTN_nw_reset_generatorClick(self);
end;

procedure TForm1.Timer_artTimer(Sender: TObject);
begin
  if train_iteration_art=n_train_cycles_art then
    begin
      Timer_art.Enabled:=false;
      Edit_n_cycles_art.text:=IntToStr(n_train_cycles_art);
    end
  else
    begin
      Timer_art.Enabled:=false;
      if CB_seed_from_S.Checked
      then MatrixCpy(tmp_elements,S_elements)
      else //randomToMatrix(S_elements,0,1); //sigmoid
           randomToMatrix(S_elements,-1,1); //tanh
      Forward_all;
      BackTraceError_all;
      BackTracetrain_generator;

      train_iteration_art:=train_iteration_art+1;
      PB_receptorsPaint(PB_receptors);
      PB_generatorPaint(PB_generator);
      Edit_L7_out.text:=FloatToStrF(L7_out[0],ffFixed,6,4);
      Edit_n_cycles_art.text:=IntToStr(train_iteration_art)+
                                    ' / '+
                                    IntToStr(n_train_cycles_art);
      Timer_art.Enabled:=true;
    end;
end;

procedure TForm1.Timer_discriminatorTimer(Sender: TObject);
var tmp_target:real;
begin
  if train_iteration_discriminator=n_train_cycles_discriminator*length(samples) then
    begin
      Timer_discriminator.Enabled:=false;
      Edit_n_cycles_Discriminator.text:=IntToStr(n_train_cycles_discriminator);
    end
  else
    begin
      Timer_discriminator.Enabled:=false;

      tmp_target:=target_discriminator;

      target_discriminator:=0.0;
      //randomToMatrix(S_elements,0,1); //sigmoid
      randomToMatrix(S_elements,-1,1); //tanh
      PB_receptorsPaint(PB_receptors);
      Forward_Discriminator;
      BackTraceError_Discriminator;
      BackTracetrain_Discriminator;

      target_discriminator:=0.95+random*0.05;
      valueToMatrix(S_elements,1);
      PB_receptorsPaint(PB_receptors);
      Forward_Discriminator;
      BackTraceError_Discriminator;
      BackTracetrain_Discriminator;

      target_discriminator:=tmp_target+(2.0*random-1.0)*target_discriminator_delta;
      MatrixCpy(samples[shuffle[train_iteration_discriminator]],S_elements);
      Forward_Discriminator;
      BackTraceError_Discriminator;
      BackTracetrain_Discriminator;

      target_discriminator:=tmp_target;

      train_iteration_discriminator:=train_iteration_discriminator+1;
      PB_receptorsPaint(PB_receptors);
      Edit_L7_out.text:=FloatToStrF(L7_out[0],ffFixed,6,4);
      Edit_n_cycles_Discriminator.text:=IntToStr(train_iteration_discriminator)+
                                        ' / '+
                               IntToStr(n_train_cycles_discriminator*length(samples));
      Timer_discriminator.Enabled:=true;
    end;
end;

procedure TForm1.Timer_generatorTimer(Sender: TObject);
begin
  if train_iteration_generator=n_train_cycles_generator*length(samples) then
    begin
      Timer_generator.Enabled:=false;
      Edit_n_cycles_generator.text:=IntToStr(n_train_cycles_generator);
    end
  else
    begin
      Timer_generator.Enabled:=false;
      MatrixCpy(samples[shuffle[train_iteration_generator]],Target_elements);
      MatrixCpy(samples[shuffle[train_iteration_generator]],S_elements);
      //noiseToMatrix(S_elements,0.05);
      Forward_generator;
      BackTraceError_generator;
      BackTracetrain_generator;

      train_iteration_generator:=train_iteration_generator+1;
      PB_receptorsPaint(PB_receptors);
      PB_generatorPaint(PB_generator);
      Edit_n_cycles_generator.text:=IntToStr(train_iteration_generator)+
                                ' / '+
                                IntToStr(n_train_cycles_generator*length(samples));
      Timer_generator.Enabled:=true;
    end;
end;

procedure TForm1.BTN_nw_reset_discriminatorClick(Sender: TObject);
begin
     n_L4:=StrToInt(Edit_N_L4.text);
     n_L5:=StrToInt(Edit_N_L5.text);
     n_L6:=StrToInt(Edit_N_L6.text);
     n_L7:=1;

     nw_discriminator_allocation;

     randomToMatrix(L4_w,-0.01,0.01); // +/- 0.3 sigmoid; +/-0.01 tanh
     randomToMatrix(L5_w,-0.01,0.01); // +/- 0.3 sigmoid; +/-0.01 tanh
     randomToMatrix(L6_w,-0.01,0.01); // +/- 0.3 sigmoid; +/-0.01 tanh
     randomToMatrix(L7_w,-0.01,0.01); // +/- 0.3 sigmoid; +/-0.01 tanh

     Forward_Discriminator;
     Edit_L7_out.text:=FloatToStrF(L7_out[0],ffFixed,6,4);
end;

procedure TForm1.BTN_nw_reset_generatorClick(Sender: TObject);
begin
  n_L1:=StrtoInt(Edit_N_L1.text);
  n_L2:=StrToInt(Edit_N_L2.text);
  n_L3:=s_width*s_height;
  Edit_N_L3.text:=IntToStr(n_L3);

  nw_generator_allocation;

  randomToMatrix(L1_w,-0.01,0.01); // +/- 0.3 sigmoid; +/-0.01 tanh
  randomToMatrix(L2_w,-0.01,0.01); // +/- 0.3 sigmoid; +/-0.01 tanh
  randomToMatrix(L3_w,-0.01,0.01); // +/- 0.3 sigmoid; +/-0.01 tanh

  Forward_generator;
  PB_generatorPaint(PB_generator);
end;

procedure TForm1.BTN_nw_saveClick(Sender: TObject);
var filename:String; f:TextFile; i,k:integer;
begin
  filename:='gen_nw.txt';
  AssignFile(f,filename);
  rewrite(f);
  writeln(f,7);
  writeln(f,n_L1,' ',n_generator_inputs);
  writeln(f,n_L2,' ',n_L1);
  writeln(f,n_L3,' ',n_L2);
  writeln(f,n_L4,' ',n_L3);
  writeln(f,n_L5,' ',n_L4);
  writeln(f,n_L6,' ',n_L5);
  writeln(f,n_L7,' ',n_L6);

  for i:=0 to n_L1-1 do
  begin
      for k:=0 to n_generator_inputs-1 do write(f,L1_w[i,k]:10:6);
      writeln(f);
  end;

  for i:=0 to n_L2-1 do
  begin
      for k:=0 to n_L1-1 do write(f,L2_w[i,k]:10:6);
      writeln(f);
  end;

  for i:=0 to n_L3-1 do
  begin
      for k:=0 to n_L2-1 do write(f,L3_w[i,k]:10:6);
      writeln(f);
  end;

  for i:=0 to n_L4-1 do
  begin
      for k:=0 to n_L3-1 do write(f,L4_w[i,k]:10:6);
      writeln(f);
  end;

  for i:=0 to n_L5-1 do
  begin
      for k:=0 to n_L4-1 do write(f,L5_w[i,k]:10:6);
      writeln(f);
  end;

  for i:=0 to n_L6-1 do
  begin
      for k:=0 to n_L5-1 do write(f,L6_w[i,k]:10:6);
      writeln(f);
  end;

  for i:=0 to n_L7-1 do
  begin
      for k:=0 to n_L6-1 do write(f,L7_w[i,k]:10:6);
      writeln(f);
  end;

  CloseFile(f);
end;

procedure TForm1.BTN_shake_generatorClick(Sender: TObject);
begin
  shakeMatrix(L1_w,alpha_shake_generator);
  shakeMatrix(L2_w,alpha_shake_generator);
  shakeMatrix(L3_w,alpha_shake_generator);

  Forward_generator;
  PB_generatorPaint(PB_generator);
end;

procedure TForm1.BTN_shake_discriminatorClick(Sender: TObject);
begin
  shakeMatrix(L4_w,alpha_shake_discriminator);
  shakeMatrix(L5_w,alpha_shake_discriminator);
  shakeMatrix(L6_w,alpha_shake_discriminator);
  shakeMatrix(L7_w,alpha_shake_discriminator);

  Forward_Discriminator;
  Edit_L7_out.text:=FloatToStrF(L7_out[0],ffFixed,6,4);
end;

procedure TForm1.BTN_stop_train_artClick(Sender: TObject);
begin
  Timer_art.Enabled:=false;
  train_iteration_art:=n_train_cycles_art;
  Timer_art.Enabled:=true;
end;

procedure TForm1.BTN_stop_train_discriminatorClick(Sender: TObject);
begin
  Timer_discriminator.Enabled:=false;
  train_iteration_discriminator:=n_train_cycles_discriminator*length(samples);
  Timer_discriminator.Enabled:=true;
end;

procedure TForm1.BTN_samples_loadClick(Sender: TObject);
var VideofileName: String;
    CameraPicture:TPicture;
    cell_x,cell_y,x,y,src_bpp:integer;
    src_ptr:PByte;
    R,G,B:word;
    dx,dy:real;
    C:real;
    k,n_samples:integer;
begin
  Timer_art.Enabled:=false;
  Timer_discriminator.Enabled:=false;
  if OpenPictureDialog.execute then
  begin
    n_samples:=OpenPictureDialog.Files.Count;
    SetLength(samples,n_samples,s_width,s_height);
    CameraPicture:=TPicture.Create;
    for k:=0 to n_samples-1 do
    begin
      VideofileName:=OpenPictureDialog.Files[k];
      CameraPicture.LoadFromFile(VideofileName);

      src_ptr:=CameraPicture.Bitmap.RawImage.Data;
      src_bpp:=CameraPicture.Bitmap.RawImage.Description.BitsPerPixel div 8;
      for y:=0 to CameraPicture.Bitmap.height-1 do
      for x:=0 to CameraPicture.Bitmap.width-1 do
      begin
        R:=(src_ptr+2)^; G:=(src_ptr+1)^; B:=src_ptr^; inc(src_ptr,src_bpp);
        img_buffer[x,y]:=(R+G+B)/(256.0*3.0);
      end;

      dx:=CameraPicture.Bitmap.Width/s_width;
      dy:=CameraPicture.Bitmap.Height/s_height;
      for cell_x:=0 to s_width-1 do
      for cell_y:=0 to s_height-1 do
      begin
        C:=0;
        for x:=0 to trunc(dx) do
        for y:=0 to trunc(dy) do
          C:=C+img_buffer[trunc(x+cell_x*dx),trunc(y+cell_y*dy)];
        samples[k,cell_x,cell_y]:=C/((trunc(dx)+1)*(trunc(dy)+1)); //sigmoid
        samples[k,cell_x,cell_y]:=2.0*(samples[k,cell_x,cell_y]-0.5); //tahh
      end;
    end;
    CameraPicture.Free;
    SB_samples.Position:=0;
    SB_samples.max:=length(samples)-1;
    SB_samplesChange(self);
  end;
end;

procedure TForm1.BTN_out_gen_to_receptorsClick(Sender: TObject);
begin
  MatrixCpy(Z_elements,S_elements);
  PB_receptorsPaint(PB_receptors);
end;

procedure TForm1.BTN_forward_discriminatorClick(Sender: TObject);
begin
  Forward_Discriminator;
  Edit_L7_out.text:=FloatToStrF(L7_out[0],ffFixed,6,4);
end;

procedure TForm1.BTN_forward_allClick(Sender: TObject);
begin
  Forward_all;
  PB_receptorsPaint(PB_receptors);
  PB_generatorPaint(PB_generator);
  Edit_L7_out.text:=FloatToStrF(L7_out[0],ffFixed,6,4);
end;

procedure TForm1.BTN_out_gen_saveClick(Sender: TObject);
begin
  generatorBitmap.SaveToFile('art_out.bmp');
end;

procedure TForm1.BTN_stop_train_generatorClick(Sender: TObject);
begin
  Timer_generator.Enabled:=false;
  train_iteration_generator:=n_train_cycles_generator*length(samples);
  Timer_generator.Enabled:=true;
end;

procedure TForm1.BTN_s_clear1Click(Sender: TObject);
begin
  valueToMatrix(S_elements,1);
  PB_receptorsPaint(PB_receptors);
end;

procedure TForm1.BTN_s_clear2Click(Sender: TObject);
begin
  //valueToMatrix(S_elements,0.5); //sigmoid
  valueToMatrix(S_elements,0); //tanh
  PB_receptorsPaint(PB_receptors);
end;

procedure TForm1.BTN_s_noiseClick(Sender: TObject);
begin
  noiseToMatrix(S_elements,0.05);
  PB_receptorsPaint(PB_receptors);
end;

procedure TForm1.BTN_train_discriminatorClick(Sender: TObject);
var k,n_samples:integer;
begin
  Timer_generator.Enabled:=false;
  Timer_art.Enabled:=false;
  Timer_discriminator.Enabled:=false;
  n_samples:=length(samples);
  target_discriminator:=StrToFloat(Edit_target_Discriminator.text);
  target_discriminator_delta:=StrToFloat(Edit_target_Discriminator_delta.text);
  n_train_cycles_discriminator:=StrToInt(Edit_n_cycles_Discriminator.text);
  SetLength(shuffle,n_samples*n_train_cycles_discriminator);
  for k:=0 to n_samples*n_train_cycles_discriminator-1 do shuffle[k]:=k mod n_samples;
  shuffleIntegerVector(shuffle);

  train_iteration_discriminator:=0;
  Timer_discriminator.Enabled:=True;
end;

procedure TForm1.BTN_train_artClick(Sender: TObject);
begin
  Timer_generator.Enabled:=false;
  Timer_art.Enabled:=false;
  Timer_discriminator.Enabled:=false;

  MatrixCpy(S_elements,tmp_elements);
  target_art:=StrToFloat(Edit_target_art.text);
  n_train_cycles_art:=StrToInt(Edit_n_cycles_art.Text);
  train_iteration_art:=0;
  Timer_art.Enabled:=True;
end;

procedure TForm1.BTN_train_generatorClick(Sender: TObject);
var k,n_samples:integer;
begin
  Timer_generator.Enabled:=false;
  Timer_art.Enabled:=false;
  Timer_discriminator.Enabled:=false;
  n_samples:=length(samples);
  n_train_cycles_generator:=StrToInt(Edit_n_cycles_generator.text);
  SetLength(shuffle,n_samples*n_train_cycles_generator);
  for k:=0 to n_samples*n_train_cycles_generator-1 do shuffle[k]:=k mod n_samples;
  shuffleIntegerVector(shuffle);

  train_iteration_generator:=0;
  Timer_generator.Enabled:=True;
end;

procedure TForm1.BTN_nw_loadClick(Sender: TObject);
var filename:String; f:TextFile; i,k,tmp:integer;
begin
  filename:='gen_nw.txt';
  AssignFile(f,filename);
  reset(f);
  read(f,tmp);
  read(f,n_L1); read(f,tmp);
  read(f,n_L2); read(f,tmp);
  read(f,n_L3); read(f,tmp);
  read(f,n_L4); read(f,tmp);
  read(f,n_L5); read(f,tmp);
  read(f,n_L6); read(f,tmp);
  read(f,n_L7); read(f,tmp);
  Edit_N_L1.Text:=IntToStr(n_L1);
  Edit_N_L2.Text:=IntToStr(n_L2);
  Edit_N_L3.Text:=IntToStr(n_L3);
  Edit_N_L4.Text:=IntToStr(n_L4);
  Edit_N_L5.Text:=IntToStr(n_L5);
  Edit_N_L6.Text:=IntToStr(n_L6);

  nw_discriminator_allocation;
  nw_generator_allocation;

  for i:=0 to n_L1-1 do
    for k:=0 to n_generator_inputs-1 do read(f,L1_w[i,k]);
  for i:=0 to n_L2-1 do
    for k:=0 to n_L1-1 do read(f,L2_w[i,k]);
  for i:=0 to n_L3-1 do
    for k:=0 to n_L2-1 do read(f,L3_w[i,k]);
  for i:=0 to n_L4-1 do
    for k:=0 to n_L3-1 do read(f,L4_w[i,k]);
  for i:=0 to n_L5-1 do
    for k:=0 to n_L4-1 do read(f,L5_w[i,k]);
  for i:=0 to n_L6-1 do
    for k:=0 to n_L5-1 do read(f,L6_w[i,k]);
  for i:=0 to n_L7-1 do
    for k:=0 to n_L6-1 do read(f,L7_w[i,k]);
  CloseFile(f);

  noiseToMatrix(S_elements,1);
  PB_receptorsPaint(PB_receptors);
  BTN_forward_allClick(self);
  PB_generatorPaint(PB_generator);
  Edit_L7_out.text:=FloatToStrF(L7_out[0],ffFixed,6,4);
end;

procedure TForm1.BTN_s_clear0Click(Sender: TObject);
begin
  //valueToMatrix(S_elements,0); //sigmoid
  valueToMatrix(S_elements,-1); //tanh
  PB_receptorsPaint(PB_receptors);
end;

procedure TForm1.BTN_s_loadClick(Sender: TObject);
var VideofileName: String;
    picture:TPicture;
    cell_x,cell_y,x,y,src_bpp:integer;
    src_ptr:PByte;
    R,G,B:word;
    dx,dy:real;
    C:real;
begin
  if OpenPictureDialog.execute then
  begin
      picture:=TPicture.Create;
      VideofileName:=OpenPictureDialog.FileName;
      picture.LoadFromFile(VideofileName);

      src_ptr:=picture.Bitmap.RawImage.Data;
      src_bpp:=picture.Bitmap.RawImage.Description.BitsPerPixel div 8;
      for y:=0 to picture.Bitmap.height-1 do
      for x:=0 to picture.Bitmap.width-1 do
      begin
          R:=(src_ptr+2)^; G:=(src_ptr+1)^; B:=src_ptr^; inc(src_ptr,src_bpp);
          img_buffer[x,y]:=(R+G+B)/(256.0*3.0);
      end;

      dx:=picture.Bitmap.Width/s_width;
      dy:=picture.Bitmap.Height/s_height;
      for cell_x:=0 to s_width-1 do
      for cell_y:=0 to s_height-1 do
      begin
          C:=0;
          for x:=0 to trunc(dx) do
          for y:=0 to trunc(dy) do
            C:=C+img_buffer[trunc(x+cell_x*dx),trunc(y+cell_y*dy)];
          S_elements[cell_x,cell_y]:=C/((trunc(dx)+1)*(trunc(dy)+1)); //sigmoid
          S_elements[cell_x,cell_y]:=2.0*(S_elements[cell_x,cell_y]-0.5); //tanh
      end;
      PB_receptorsPaint(PB_receptors);
      picture.Free;
  end;
end;

procedure TForm1.BTN_s_randomClick(Sender: TObject);
begin
  //randomToMatrix(S_elements,0,1); //sigmoid
  randomToMatrix(S_elements,-1,1); //tanh
  PB_receptorsPaint(PB_receptors);
end;

procedure TForm1.BTN_s_saveClick(Sender: TObject);
begin
  receptorsBitmap.SaveToFile('art_receptors.bmp');
end;

procedure TForm1.CB_contrastChange(Sender: TObject);
begin
  Edit_contrast.ReadOnly:=CB_contrast.Checked;
  PB_generatorPaint(PB_generator);
end;

procedure TForm1.CB_autolevelChange(Sender: TObject);
begin
  PB_receptorsPaint(PB_receptors);
  PB_generatorPaint(PB_generator);
end;

end.

