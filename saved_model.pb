??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
?
conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_11/kernel
}
$conv2d_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_11/kernel*&
_output_shapes
:*
dtype0
t
conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_11/bias
m
"conv2d_11/bias/Read/ReadVariableOpReadVariableOpconv2d_11/bias*
_output_shapes
:*
dtype0
?
conv2d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_12/kernel
}
$conv2d_12/kernel/Read/ReadVariableOpReadVariableOpconv2d_12/kernel*&
_output_shapes
: *
dtype0
t
conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_12/bias
m
"conv2d_12/bias/Read/ReadVariableOpReadVariableOpconv2d_12/bias*
_output_shapes
: *
dtype0
?
conv2d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_13/kernel
}
$conv2d_13/kernel/Read/ReadVariableOpReadVariableOpconv2d_13/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_13/bias
m
"conv2d_13/bias/Read/ReadVariableOpReadVariableOpconv2d_13/bias*
_output_shapes
:@*
dtype0
|
dense_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_27/kernel
u
#dense_27/kernel/Read/ReadVariableOpReadVariableOpdense_27/kernel* 
_output_shapes
:
??*
dtype0
s
dense_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_27/bias
l
!dense_27/bias/Read/ReadVariableOpReadVariableOpdense_27/bias*
_output_shapes	
:?*
dtype0
{
dense_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@* 
shared_namedense_28/kernel
t
#dense_28/kernel/Read/ReadVariableOpReadVariableOpdense_28/kernel*
_output_shapes
:	?@*
dtype0
r
dense_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_28/bias
k
!dense_28/bias/Read/ReadVariableOpReadVariableOpdense_28/bias*
_output_shapes
:@*
dtype0
z
dense_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ * 
shared_namedense_29/kernel
s
#dense_29/kernel/Read/ReadVariableOpReadVariableOpdense_29/kernel*
_output_shapes

:@ *
dtype0
r
dense_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_29/bias
k
!dense_29/bias/Read/ReadVariableOpReadVariableOpdense_29/bias*
_output_shapes
: *
dtype0
z
dense_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_30/kernel
s
#dense_30/kernel/Read/ReadVariableOpReadVariableOpdense_30/kernel*
_output_shapes

: *
dtype0
r
dense_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_30/bias
k
!dense_30/bias/Read/ReadVariableOpReadVariableOpdense_30/bias*
_output_shapes
:*
dtype0
z
dense_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_31/kernel
s
#dense_31/kernel/Read/ReadVariableOpReadVariableOpdense_31/kernel*
_output_shapes

: *
dtype0
r
dense_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_31/bias
k
!dense_31/bias/Read/ReadVariableOpReadVariableOpdense_31/bias*
_output_shapes
:*
dtype0
z
dense_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_32/kernel
s
#dense_32/kernel/Read/ReadVariableOpReadVariableOpdense_32/kernel*
_output_shapes

: *
dtype0
r
dense_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_32/bias
k
!dense_32/bias/Read/ReadVariableOpReadVariableOpdense_32/bias*
_output_shapes
:*
dtype0
z
dense_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_33/kernel
s
#dense_33/kernel/Read/ReadVariableOpReadVariableOpdense_33/kernel*
_output_shapes

: *
dtype0
r
dense_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_33/bias
k
!dense_33/bias/Read/ReadVariableOpReadVariableOpdense_33/bias*
_output_shapes
:*
dtype0
z
dense_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_34/kernel
s
#dense_34/kernel/Read/ReadVariableOpReadVariableOpdense_34/kernel*
_output_shapes

: *
dtype0
r
dense_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_34/bias
k
!dense_34/bias/Read/ReadVariableOpReadVariableOpdense_34/bias*
_output_shapes
:*
dtype0
z
dense_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_35/kernel
s
#dense_35/kernel/Read/ReadVariableOpReadVariableOpdense_35/kernel*
_output_shapes

: *
dtype0
r
dense_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_35/bias
k
!dense_35/bias/Read/ReadVariableOpReadVariableOpdense_35/bias*
_output_shapes
:*
dtype0
|
training_8/Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *%
shared_nametraining_8/Adam/iter
u
(training_8/Adam/iter/Read/ReadVariableOpReadVariableOptraining_8/Adam/iter*
_output_shapes
: *
dtype0	
?
training_8/Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nametraining_8/Adam/beta_1
y
*training_8/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining_8/Adam/beta_1*
_output_shapes
: *
dtype0
?
training_8/Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nametraining_8/Adam/beta_2
y
*training_8/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining_8/Adam/beta_2*
_output_shapes
: *
dtype0
~
training_8/Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nametraining_8/Adam/decay
w
)training_8/Adam/decay/Read/ReadVariableOpReadVariableOptraining_8/Adam/decay*
_output_shapes
: *
dtype0
?
training_8/Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nametraining_8/Adam/learning_rate
?
1training_8/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining_8/Adam/learning_rate*
_output_shapes
: *
dtype0
d
total_48VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_48
]
total_48/Read/ReadVariableOpReadVariableOptotal_48*
_output_shapes
: *
dtype0
d
count_48VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_48
]
count_48/Read/ReadVariableOpReadVariableOpcount_48*
_output_shapes
: *
dtype0
d
total_49VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_49
]
total_49/Read/ReadVariableOpReadVariableOptotal_49*
_output_shapes
: *
dtype0
d
count_49VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_49
]
count_49/Read/ReadVariableOpReadVariableOpcount_49*
_output_shapes
: *
dtype0
d
total_50VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_50
]
total_50/Read/ReadVariableOpReadVariableOptotal_50*
_output_shapes
: *
dtype0
d
count_50VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_50
]
count_50/Read/ReadVariableOpReadVariableOpcount_50*
_output_shapes
: *
dtype0
d
total_51VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_51
]
total_51/Read/ReadVariableOpReadVariableOptotal_51*
_output_shapes
: *
dtype0
d
count_51VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_51
]
count_51/Read/ReadVariableOpReadVariableOpcount_51*
_output_shapes
: *
dtype0
d
total_52VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_52
]
total_52/Read/ReadVariableOpReadVariableOptotal_52*
_output_shapes
: *
dtype0
d
count_52VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_52
]
count_52/Read/ReadVariableOpReadVariableOpcount_52*
_output_shapes
: *
dtype0
d
total_53VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_53
]
total_53/Read/ReadVariableOpReadVariableOptotal_53*
_output_shapes
: *
dtype0
d
count_53VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_53
]
count_53/Read/ReadVariableOpReadVariableOpcount_53*
_output_shapes
: *
dtype0
?
"training_8/Adam/conv2d_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"training_8/Adam/conv2d_11/kernel/m
?
6training_8/Adam/conv2d_11/kernel/m/Read/ReadVariableOpReadVariableOp"training_8/Adam/conv2d_11/kernel/m*&
_output_shapes
:*
dtype0
?
 training_8/Adam/conv2d_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" training_8/Adam/conv2d_11/bias/m
?
4training_8/Adam/conv2d_11/bias/m/Read/ReadVariableOpReadVariableOp training_8/Adam/conv2d_11/bias/m*
_output_shapes
:*
dtype0
?
"training_8/Adam/conv2d_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"training_8/Adam/conv2d_12/kernel/m
?
6training_8/Adam/conv2d_12/kernel/m/Read/ReadVariableOpReadVariableOp"training_8/Adam/conv2d_12/kernel/m*&
_output_shapes
: *
dtype0
?
 training_8/Adam/conv2d_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" training_8/Adam/conv2d_12/bias/m
?
4training_8/Adam/conv2d_12/bias/m/Read/ReadVariableOpReadVariableOp training_8/Adam/conv2d_12/bias/m*
_output_shapes
: *
dtype0
?
"training_8/Adam/conv2d_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*3
shared_name$"training_8/Adam/conv2d_13/kernel/m
?
6training_8/Adam/conv2d_13/kernel/m/Read/ReadVariableOpReadVariableOp"training_8/Adam/conv2d_13/kernel/m*&
_output_shapes
: @*
dtype0
?
 training_8/Adam/conv2d_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" training_8/Adam/conv2d_13/bias/m
?
4training_8/Adam/conv2d_13/bias/m/Read/ReadVariableOpReadVariableOp training_8/Adam/conv2d_13/bias/m*
_output_shapes
:@*
dtype0
?
!training_8/Adam/dense_27/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*2
shared_name#!training_8/Adam/dense_27/kernel/m
?
5training_8/Adam/dense_27/kernel/m/Read/ReadVariableOpReadVariableOp!training_8/Adam/dense_27/kernel/m* 
_output_shapes
:
??*
dtype0
?
training_8/Adam/dense_27/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*0
shared_name!training_8/Adam/dense_27/bias/m
?
3training_8/Adam/dense_27/bias/m/Read/ReadVariableOpReadVariableOptraining_8/Adam/dense_27/bias/m*
_output_shapes	
:?*
dtype0
?
!training_8/Adam/dense_28/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*2
shared_name#!training_8/Adam/dense_28/kernel/m
?
5training_8/Adam/dense_28/kernel/m/Read/ReadVariableOpReadVariableOp!training_8/Adam/dense_28/kernel/m*
_output_shapes
:	?@*
dtype0
?
training_8/Adam/dense_28/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!training_8/Adam/dense_28/bias/m
?
3training_8/Adam/dense_28/bias/m/Read/ReadVariableOpReadVariableOptraining_8/Adam/dense_28/bias/m*
_output_shapes
:@*
dtype0
?
!training_8/Adam/dense_29/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *2
shared_name#!training_8/Adam/dense_29/kernel/m
?
5training_8/Adam/dense_29/kernel/m/Read/ReadVariableOpReadVariableOp!training_8/Adam/dense_29/kernel/m*
_output_shapes

:@ *
dtype0
?
training_8/Adam/dense_29/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!training_8/Adam/dense_29/bias/m
?
3training_8/Adam/dense_29/bias/m/Read/ReadVariableOpReadVariableOptraining_8/Adam/dense_29/bias/m*
_output_shapes
: *
dtype0
?
!training_8/Adam/dense_30/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *2
shared_name#!training_8/Adam/dense_30/kernel/m
?
5training_8/Adam/dense_30/kernel/m/Read/ReadVariableOpReadVariableOp!training_8/Adam/dense_30/kernel/m*
_output_shapes

: *
dtype0
?
training_8/Adam/dense_30/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!training_8/Adam/dense_30/bias/m
?
3training_8/Adam/dense_30/bias/m/Read/ReadVariableOpReadVariableOptraining_8/Adam/dense_30/bias/m*
_output_shapes
:*
dtype0
?
!training_8/Adam/dense_31/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *2
shared_name#!training_8/Adam/dense_31/kernel/m
?
5training_8/Adam/dense_31/kernel/m/Read/ReadVariableOpReadVariableOp!training_8/Adam/dense_31/kernel/m*
_output_shapes

: *
dtype0
?
training_8/Adam/dense_31/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!training_8/Adam/dense_31/bias/m
?
3training_8/Adam/dense_31/bias/m/Read/ReadVariableOpReadVariableOptraining_8/Adam/dense_31/bias/m*
_output_shapes
:*
dtype0
?
!training_8/Adam/dense_32/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *2
shared_name#!training_8/Adam/dense_32/kernel/m
?
5training_8/Adam/dense_32/kernel/m/Read/ReadVariableOpReadVariableOp!training_8/Adam/dense_32/kernel/m*
_output_shapes

: *
dtype0
?
training_8/Adam/dense_32/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!training_8/Adam/dense_32/bias/m
?
3training_8/Adam/dense_32/bias/m/Read/ReadVariableOpReadVariableOptraining_8/Adam/dense_32/bias/m*
_output_shapes
:*
dtype0
?
!training_8/Adam/dense_33/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *2
shared_name#!training_8/Adam/dense_33/kernel/m
?
5training_8/Adam/dense_33/kernel/m/Read/ReadVariableOpReadVariableOp!training_8/Adam/dense_33/kernel/m*
_output_shapes

: *
dtype0
?
training_8/Adam/dense_33/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!training_8/Adam/dense_33/bias/m
?
3training_8/Adam/dense_33/bias/m/Read/ReadVariableOpReadVariableOptraining_8/Adam/dense_33/bias/m*
_output_shapes
:*
dtype0
?
!training_8/Adam/dense_34/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *2
shared_name#!training_8/Adam/dense_34/kernel/m
?
5training_8/Adam/dense_34/kernel/m/Read/ReadVariableOpReadVariableOp!training_8/Adam/dense_34/kernel/m*
_output_shapes

: *
dtype0
?
training_8/Adam/dense_34/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!training_8/Adam/dense_34/bias/m
?
3training_8/Adam/dense_34/bias/m/Read/ReadVariableOpReadVariableOptraining_8/Adam/dense_34/bias/m*
_output_shapes
:*
dtype0
?
!training_8/Adam/dense_35/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *2
shared_name#!training_8/Adam/dense_35/kernel/m
?
5training_8/Adam/dense_35/kernel/m/Read/ReadVariableOpReadVariableOp!training_8/Adam/dense_35/kernel/m*
_output_shapes

: *
dtype0
?
training_8/Adam/dense_35/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!training_8/Adam/dense_35/bias/m
?
3training_8/Adam/dense_35/bias/m/Read/ReadVariableOpReadVariableOptraining_8/Adam/dense_35/bias/m*
_output_shapes
:*
dtype0
?
"training_8/Adam/conv2d_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"training_8/Adam/conv2d_11/kernel/v
?
6training_8/Adam/conv2d_11/kernel/v/Read/ReadVariableOpReadVariableOp"training_8/Adam/conv2d_11/kernel/v*&
_output_shapes
:*
dtype0
?
 training_8/Adam/conv2d_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" training_8/Adam/conv2d_11/bias/v
?
4training_8/Adam/conv2d_11/bias/v/Read/ReadVariableOpReadVariableOp training_8/Adam/conv2d_11/bias/v*
_output_shapes
:*
dtype0
?
"training_8/Adam/conv2d_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"training_8/Adam/conv2d_12/kernel/v
?
6training_8/Adam/conv2d_12/kernel/v/Read/ReadVariableOpReadVariableOp"training_8/Adam/conv2d_12/kernel/v*&
_output_shapes
: *
dtype0
?
 training_8/Adam/conv2d_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" training_8/Adam/conv2d_12/bias/v
?
4training_8/Adam/conv2d_12/bias/v/Read/ReadVariableOpReadVariableOp training_8/Adam/conv2d_12/bias/v*
_output_shapes
: *
dtype0
?
"training_8/Adam/conv2d_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*3
shared_name$"training_8/Adam/conv2d_13/kernel/v
?
6training_8/Adam/conv2d_13/kernel/v/Read/ReadVariableOpReadVariableOp"training_8/Adam/conv2d_13/kernel/v*&
_output_shapes
: @*
dtype0
?
 training_8/Adam/conv2d_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" training_8/Adam/conv2d_13/bias/v
?
4training_8/Adam/conv2d_13/bias/v/Read/ReadVariableOpReadVariableOp training_8/Adam/conv2d_13/bias/v*
_output_shapes
:@*
dtype0
?
!training_8/Adam/dense_27/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*2
shared_name#!training_8/Adam/dense_27/kernel/v
?
5training_8/Adam/dense_27/kernel/v/Read/ReadVariableOpReadVariableOp!training_8/Adam/dense_27/kernel/v* 
_output_shapes
:
??*
dtype0
?
training_8/Adam/dense_27/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*0
shared_name!training_8/Adam/dense_27/bias/v
?
3training_8/Adam/dense_27/bias/v/Read/ReadVariableOpReadVariableOptraining_8/Adam/dense_27/bias/v*
_output_shapes	
:?*
dtype0
?
!training_8/Adam/dense_28/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*2
shared_name#!training_8/Adam/dense_28/kernel/v
?
5training_8/Adam/dense_28/kernel/v/Read/ReadVariableOpReadVariableOp!training_8/Adam/dense_28/kernel/v*
_output_shapes
:	?@*
dtype0
?
training_8/Adam/dense_28/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!training_8/Adam/dense_28/bias/v
?
3training_8/Adam/dense_28/bias/v/Read/ReadVariableOpReadVariableOptraining_8/Adam/dense_28/bias/v*
_output_shapes
:@*
dtype0
?
!training_8/Adam/dense_29/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *2
shared_name#!training_8/Adam/dense_29/kernel/v
?
5training_8/Adam/dense_29/kernel/v/Read/ReadVariableOpReadVariableOp!training_8/Adam/dense_29/kernel/v*
_output_shapes

:@ *
dtype0
?
training_8/Adam/dense_29/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!training_8/Adam/dense_29/bias/v
?
3training_8/Adam/dense_29/bias/v/Read/ReadVariableOpReadVariableOptraining_8/Adam/dense_29/bias/v*
_output_shapes
: *
dtype0
?
!training_8/Adam/dense_30/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *2
shared_name#!training_8/Adam/dense_30/kernel/v
?
5training_8/Adam/dense_30/kernel/v/Read/ReadVariableOpReadVariableOp!training_8/Adam/dense_30/kernel/v*
_output_shapes

: *
dtype0
?
training_8/Adam/dense_30/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!training_8/Adam/dense_30/bias/v
?
3training_8/Adam/dense_30/bias/v/Read/ReadVariableOpReadVariableOptraining_8/Adam/dense_30/bias/v*
_output_shapes
:*
dtype0
?
!training_8/Adam/dense_31/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *2
shared_name#!training_8/Adam/dense_31/kernel/v
?
5training_8/Adam/dense_31/kernel/v/Read/ReadVariableOpReadVariableOp!training_8/Adam/dense_31/kernel/v*
_output_shapes

: *
dtype0
?
training_8/Adam/dense_31/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!training_8/Adam/dense_31/bias/v
?
3training_8/Adam/dense_31/bias/v/Read/ReadVariableOpReadVariableOptraining_8/Adam/dense_31/bias/v*
_output_shapes
:*
dtype0
?
!training_8/Adam/dense_32/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *2
shared_name#!training_8/Adam/dense_32/kernel/v
?
5training_8/Adam/dense_32/kernel/v/Read/ReadVariableOpReadVariableOp!training_8/Adam/dense_32/kernel/v*
_output_shapes

: *
dtype0
?
training_8/Adam/dense_32/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!training_8/Adam/dense_32/bias/v
?
3training_8/Adam/dense_32/bias/v/Read/ReadVariableOpReadVariableOptraining_8/Adam/dense_32/bias/v*
_output_shapes
:*
dtype0
?
!training_8/Adam/dense_33/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *2
shared_name#!training_8/Adam/dense_33/kernel/v
?
5training_8/Adam/dense_33/kernel/v/Read/ReadVariableOpReadVariableOp!training_8/Adam/dense_33/kernel/v*
_output_shapes

: *
dtype0
?
training_8/Adam/dense_33/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!training_8/Adam/dense_33/bias/v
?
3training_8/Adam/dense_33/bias/v/Read/ReadVariableOpReadVariableOptraining_8/Adam/dense_33/bias/v*
_output_shapes
:*
dtype0
?
!training_8/Adam/dense_34/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *2
shared_name#!training_8/Adam/dense_34/kernel/v
?
5training_8/Adam/dense_34/kernel/v/Read/ReadVariableOpReadVariableOp!training_8/Adam/dense_34/kernel/v*
_output_shapes

: *
dtype0
?
training_8/Adam/dense_34/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!training_8/Adam/dense_34/bias/v
?
3training_8/Adam/dense_34/bias/v/Read/ReadVariableOpReadVariableOptraining_8/Adam/dense_34/bias/v*
_output_shapes
:*
dtype0
?
!training_8/Adam/dense_35/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *2
shared_name#!training_8/Adam/dense_35/kernel/v
?
5training_8/Adam/dense_35/kernel/v/Read/ReadVariableOpReadVariableOp!training_8/Adam/dense_35/kernel/v*
_output_shapes

: *
dtype0
?
training_8/Adam/dense_35/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!training_8/Adam/dense_35/bias/v
?
3training_8/Adam/dense_35/bias/v/Read/ReadVariableOpReadVariableOptraining_8/Adam/dense_35/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer_with_weights-8
layer-13
layer_with_weights-9
layer-14
layer_with_weights-10
layer-15
layer_with_weights-11
layer-16
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
 	variables
!	keras_api
h

"kernel
#bias
$trainable_variables
%regularization_losses
&	variables
'	keras_api
R
(trainable_variables
)regularization_losses
*	variables
+	keras_api
h

,kernel
-bias
.trainable_variables
/regularization_losses
0	variables
1	keras_api
R
2trainable_variables
3regularization_losses
4	variables
5	keras_api
R
6trainable_variables
7regularization_losses
8	variables
9	keras_api
h

:kernel
;bias
<trainable_variables
=regularization_losses
>	variables
?	keras_api
h

@kernel
Abias
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
h

Fkernel
Gbias
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api
h

Lkernel
Mbias
Ntrainable_variables
Oregularization_losses
P	variables
Q	keras_api
h

Rkernel
Sbias
Ttrainable_variables
Uregularization_losses
V	variables
W	keras_api
h

Xkernel
Ybias
Ztrainable_variables
[regularization_losses
\	variables
]	keras_api
h

^kernel
_bias
`trainable_variables
aregularization_losses
b	variables
c	keras_api
h

dkernel
ebias
ftrainable_variables
gregularization_losses
h	variables
i	keras_api
h

jkernel
kbias
ltrainable_variables
mregularization_losses
n	variables
o	keras_api
?
piter

qbeta_1

rbeta_2
	sdecay
tlearning_ratem?m?"m?#m?,m?-m?:m?;m?@m?Am?Fm?Gm?Lm?Mm?Rm?Sm?Xm?Ym?^m?_m?dm?em?jm?km?v?v?"v?#v?,v?-v?:v?;v?@v?Av?Fv?Gv?Lv?Mv?Rv?Sv?Xv?Yv?^v?_v?dv?ev?jv?kv?
?
0
1
"2
#3
,4
-5
:6
;7
@8
A9
F10
G11
L12
M13
R14
S15
X16
Y17
^18
_19
d20
e21
j22
k23
 
?
0
1
"2
#3
,4
-5
:6
;7
@8
A9
F10
G11
L12
M13
R14
S15
X16
Y17
^18
_19
d20
e21
j22
k23
?
ulayer_metrics
trainable_variables
vnon_trainable_variables
wmetrics
regularization_losses
xlayer_regularization_losses
	variables

ylayers
 
\Z
VARIABLE_VALUEconv2d_11/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_11/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
zlayer_metrics
trainable_variables
{non_trainable_variables
|metrics
regularization_losses
}layer_regularization_losses
	variables

~layers
 
 
 
?
layer_metrics
trainable_variables
?non_trainable_variables
?metrics
regularization_losses
 ?layer_regularization_losses
 	variables
?layers
\Z
VARIABLE_VALUEconv2d_12/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_12/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1
 

"0
#1
?
?layer_metrics
$trainable_variables
?non_trainable_variables
?metrics
%regularization_losses
 ?layer_regularization_losses
&	variables
?layers
 
 
 
?
?layer_metrics
(trainable_variables
?non_trainable_variables
?metrics
)regularization_losses
 ?layer_regularization_losses
*	variables
?layers
\Z
VARIABLE_VALUEconv2d_13/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_13/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

,0
-1
 

,0
-1
?
?layer_metrics
.trainable_variables
?non_trainable_variables
?metrics
/regularization_losses
 ?layer_regularization_losses
0	variables
?layers
 
 
 
?
?layer_metrics
2trainable_variables
?non_trainable_variables
?metrics
3regularization_losses
 ?layer_regularization_losses
4	variables
?layers
 
 
 
?
?layer_metrics
6trainable_variables
?non_trainable_variables
?metrics
7regularization_losses
 ?layer_regularization_losses
8	variables
?layers
[Y
VARIABLE_VALUEdense_27/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_27/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

:0
;1
 

:0
;1
?
?layer_metrics
<trainable_variables
?non_trainable_variables
?metrics
=regularization_losses
 ?layer_regularization_losses
>	variables
?layers
[Y
VARIABLE_VALUEdense_28/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_28/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

@0
A1
 

@0
A1
?
?layer_metrics
Btrainable_variables
?non_trainable_variables
?metrics
Cregularization_losses
 ?layer_regularization_losses
D	variables
?layers
[Y
VARIABLE_VALUEdense_29/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_29/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

F0
G1
 

F0
G1
?
?layer_metrics
Htrainable_variables
?non_trainable_variables
?metrics
Iregularization_losses
 ?layer_regularization_losses
J	variables
?layers
[Y
VARIABLE_VALUEdense_30/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_30/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

L0
M1
 

L0
M1
?
?layer_metrics
Ntrainable_variables
?non_trainable_variables
?metrics
Oregularization_losses
 ?layer_regularization_losses
P	variables
?layers
[Y
VARIABLE_VALUEdense_31/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_31/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

R0
S1
 

R0
S1
?
?layer_metrics
Ttrainable_variables
?non_trainable_variables
?metrics
Uregularization_losses
 ?layer_regularization_losses
V	variables
?layers
[Y
VARIABLE_VALUEdense_32/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_32/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

X0
Y1
 

X0
Y1
?
?layer_metrics
Ztrainable_variables
?non_trainable_variables
?metrics
[regularization_losses
 ?layer_regularization_losses
\	variables
?layers
[Y
VARIABLE_VALUEdense_33/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_33/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

^0
_1
 

^0
_1
?
?layer_metrics
`trainable_variables
?non_trainable_variables
?metrics
aregularization_losses
 ?layer_regularization_losses
b	variables
?layers
\Z
VARIABLE_VALUEdense_34/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_34/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

d0
e1
 

d0
e1
?
?layer_metrics
ftrainable_variables
?non_trainable_variables
?metrics
gregularization_losses
 ?layer_regularization_losses
h	variables
?layers
\Z
VARIABLE_VALUEdense_35/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_35/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

j0
k1
 

j0
k1
?
?layer_metrics
ltrainable_variables
?non_trainable_variables
?metrics
mregularization_losses
 ?layer_regularization_losses
n	variables
?layers
SQ
VARIABLE_VALUEtraining_8/Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEtraining_8/Adam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEtraining_8/Adam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtraining_8/Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEtraining_8/Adam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
0
?0
?1
?2
?3
?4
?5
 
~
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
RP
VARIABLE_VALUEtotal_484keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_484keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
RP
VARIABLE_VALUEtotal_494keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_494keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
RP
VARIABLE_VALUEtotal_504keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_504keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
RP
VARIABLE_VALUEtotal_514keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_514keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
RP
VARIABLE_VALUEtotal_524keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_524keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
RP
VARIABLE_VALUEtotal_534keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_534keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
??
VARIABLE_VALUE"training_8/Adam/conv2d_11/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training_8/Adam/conv2d_11/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_8/Adam/conv2d_12/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training_8/Adam/conv2d_12/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_8/Adam/conv2d_13/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training_8/Adam/conv2d_13/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!training_8/Adam/dense_27/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining_8/Adam/dense_27/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!training_8/Adam/dense_28/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining_8/Adam/dense_28/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!training_8/Adam/dense_29/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining_8/Adam/dense_29/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!training_8/Adam/dense_30/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining_8/Adam/dense_30/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!training_8/Adam/dense_31/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining_8/Adam/dense_31/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!training_8/Adam/dense_32/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining_8/Adam/dense_32/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!training_8/Adam/dense_33/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining_8/Adam/dense_33/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!training_8/Adam/dense_34/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining_8/Adam/dense_34/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!training_8/Adam/dense_35/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining_8/Adam/dense_35/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_8/Adam/conv2d_11/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training_8/Adam/conv2d_11/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_8/Adam/conv2d_12/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training_8/Adam/conv2d_12/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"training_8/Adam/conv2d_13/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training_8/Adam/conv2d_13/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!training_8/Adam/dense_27/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining_8/Adam/dense_27/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!training_8/Adam/dense_28/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining_8/Adam/dense_28/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!training_8/Adam/dense_29/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining_8/Adam/dense_29/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!training_8/Adam/dense_30/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining_8/Adam/dense_30/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!training_8/Adam/dense_31/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining_8/Adam/dense_31/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!training_8/Adam/dense_32/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining_8/Adam/dense_32/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!training_8/Adam/dense_33/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining_8/Adam/dense_33/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!training_8/Adam/dense_34/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining_8/Adam/dense_34/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!training_8/Adam/dense_35/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining_8/Adam/dense_35/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_4Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_4conv2d_11/kernelconv2d_11/biasconv2d_12/kernelconv2d_12/biasconv2d_13/kernelconv2d_13/biasdense_27/kerneldense_27/biasdense_28/kerneldense_28/biasdense_29/kerneldense_29/biasdense_35/kerneldense_35/biasdense_34/kerneldense_34/biasdense_33/kerneldense_33/biasdense_32/kerneldense_32/biasdense_31/kerneldense_31/biasdense_30/kerneldense_30/bias*$
Tin
2*
Tout

2*
_collective_manager_ids
 *?
_output_shapest
r:?????????:?????????:?????????:?????????:?????????:?????????*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_11178
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?"
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_11/kernel/Read/ReadVariableOp"conv2d_11/bias/Read/ReadVariableOp$conv2d_12/kernel/Read/ReadVariableOp"conv2d_12/bias/Read/ReadVariableOp$conv2d_13/kernel/Read/ReadVariableOp"conv2d_13/bias/Read/ReadVariableOp#dense_27/kernel/Read/ReadVariableOp!dense_27/bias/Read/ReadVariableOp#dense_28/kernel/Read/ReadVariableOp!dense_28/bias/Read/ReadVariableOp#dense_29/kernel/Read/ReadVariableOp!dense_29/bias/Read/ReadVariableOp#dense_30/kernel/Read/ReadVariableOp!dense_30/bias/Read/ReadVariableOp#dense_31/kernel/Read/ReadVariableOp!dense_31/bias/Read/ReadVariableOp#dense_32/kernel/Read/ReadVariableOp!dense_32/bias/Read/ReadVariableOp#dense_33/kernel/Read/ReadVariableOp!dense_33/bias/Read/ReadVariableOp#dense_34/kernel/Read/ReadVariableOp!dense_34/bias/Read/ReadVariableOp#dense_35/kernel/Read/ReadVariableOp!dense_35/bias/Read/ReadVariableOp(training_8/Adam/iter/Read/ReadVariableOp*training_8/Adam/beta_1/Read/ReadVariableOp*training_8/Adam/beta_2/Read/ReadVariableOp)training_8/Adam/decay/Read/ReadVariableOp1training_8/Adam/learning_rate/Read/ReadVariableOptotal_48/Read/ReadVariableOpcount_48/Read/ReadVariableOptotal_49/Read/ReadVariableOpcount_49/Read/ReadVariableOptotal_50/Read/ReadVariableOpcount_50/Read/ReadVariableOptotal_51/Read/ReadVariableOpcount_51/Read/ReadVariableOptotal_52/Read/ReadVariableOpcount_52/Read/ReadVariableOptotal_53/Read/ReadVariableOpcount_53/Read/ReadVariableOp6training_8/Adam/conv2d_11/kernel/m/Read/ReadVariableOp4training_8/Adam/conv2d_11/bias/m/Read/ReadVariableOp6training_8/Adam/conv2d_12/kernel/m/Read/ReadVariableOp4training_8/Adam/conv2d_12/bias/m/Read/ReadVariableOp6training_8/Adam/conv2d_13/kernel/m/Read/ReadVariableOp4training_8/Adam/conv2d_13/bias/m/Read/ReadVariableOp5training_8/Adam/dense_27/kernel/m/Read/ReadVariableOp3training_8/Adam/dense_27/bias/m/Read/ReadVariableOp5training_8/Adam/dense_28/kernel/m/Read/ReadVariableOp3training_8/Adam/dense_28/bias/m/Read/ReadVariableOp5training_8/Adam/dense_29/kernel/m/Read/ReadVariableOp3training_8/Adam/dense_29/bias/m/Read/ReadVariableOp5training_8/Adam/dense_30/kernel/m/Read/ReadVariableOp3training_8/Adam/dense_30/bias/m/Read/ReadVariableOp5training_8/Adam/dense_31/kernel/m/Read/ReadVariableOp3training_8/Adam/dense_31/bias/m/Read/ReadVariableOp5training_8/Adam/dense_32/kernel/m/Read/ReadVariableOp3training_8/Adam/dense_32/bias/m/Read/ReadVariableOp5training_8/Adam/dense_33/kernel/m/Read/ReadVariableOp3training_8/Adam/dense_33/bias/m/Read/ReadVariableOp5training_8/Adam/dense_34/kernel/m/Read/ReadVariableOp3training_8/Adam/dense_34/bias/m/Read/ReadVariableOp5training_8/Adam/dense_35/kernel/m/Read/ReadVariableOp3training_8/Adam/dense_35/bias/m/Read/ReadVariableOp6training_8/Adam/conv2d_11/kernel/v/Read/ReadVariableOp4training_8/Adam/conv2d_11/bias/v/Read/ReadVariableOp6training_8/Adam/conv2d_12/kernel/v/Read/ReadVariableOp4training_8/Adam/conv2d_12/bias/v/Read/ReadVariableOp6training_8/Adam/conv2d_13/kernel/v/Read/ReadVariableOp4training_8/Adam/conv2d_13/bias/v/Read/ReadVariableOp5training_8/Adam/dense_27/kernel/v/Read/ReadVariableOp3training_8/Adam/dense_27/bias/v/Read/ReadVariableOp5training_8/Adam/dense_28/kernel/v/Read/ReadVariableOp3training_8/Adam/dense_28/bias/v/Read/ReadVariableOp5training_8/Adam/dense_29/kernel/v/Read/ReadVariableOp3training_8/Adam/dense_29/bias/v/Read/ReadVariableOp5training_8/Adam/dense_30/kernel/v/Read/ReadVariableOp3training_8/Adam/dense_30/bias/v/Read/ReadVariableOp5training_8/Adam/dense_31/kernel/v/Read/ReadVariableOp3training_8/Adam/dense_31/bias/v/Read/ReadVariableOp5training_8/Adam/dense_32/kernel/v/Read/ReadVariableOp3training_8/Adam/dense_32/bias/v/Read/ReadVariableOp5training_8/Adam/dense_33/kernel/v/Read/ReadVariableOp3training_8/Adam/dense_33/bias/v/Read/ReadVariableOp5training_8/Adam/dense_34/kernel/v/Read/ReadVariableOp3training_8/Adam/dense_34/bias/v/Read/ReadVariableOp5training_8/Adam/dense_35/kernel/v/Read/ReadVariableOp3training_8/Adam/dense_35/bias/v/Read/ReadVariableOpConst*f
Tin_
]2[	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_11974
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_11/kernelconv2d_11/biasconv2d_12/kernelconv2d_12/biasconv2d_13/kernelconv2d_13/biasdense_27/kerneldense_27/biasdense_28/kerneldense_28/biasdense_29/kerneldense_29/biasdense_30/kerneldense_30/biasdense_31/kerneldense_31/biasdense_32/kerneldense_32/biasdense_33/kerneldense_33/biasdense_34/kerneldense_34/biasdense_35/kerneldense_35/biastraining_8/Adam/itertraining_8/Adam/beta_1training_8/Adam/beta_2training_8/Adam/decaytraining_8/Adam/learning_ratetotal_48count_48total_49count_49total_50count_50total_51count_51total_52count_52total_53count_53"training_8/Adam/conv2d_11/kernel/m training_8/Adam/conv2d_11/bias/m"training_8/Adam/conv2d_12/kernel/m training_8/Adam/conv2d_12/bias/m"training_8/Adam/conv2d_13/kernel/m training_8/Adam/conv2d_13/bias/m!training_8/Adam/dense_27/kernel/mtraining_8/Adam/dense_27/bias/m!training_8/Adam/dense_28/kernel/mtraining_8/Adam/dense_28/bias/m!training_8/Adam/dense_29/kernel/mtraining_8/Adam/dense_29/bias/m!training_8/Adam/dense_30/kernel/mtraining_8/Adam/dense_30/bias/m!training_8/Adam/dense_31/kernel/mtraining_8/Adam/dense_31/bias/m!training_8/Adam/dense_32/kernel/mtraining_8/Adam/dense_32/bias/m!training_8/Adam/dense_33/kernel/mtraining_8/Adam/dense_33/bias/m!training_8/Adam/dense_34/kernel/mtraining_8/Adam/dense_34/bias/m!training_8/Adam/dense_35/kernel/mtraining_8/Adam/dense_35/bias/m"training_8/Adam/conv2d_11/kernel/v training_8/Adam/conv2d_11/bias/v"training_8/Adam/conv2d_12/kernel/v training_8/Adam/conv2d_12/bias/v"training_8/Adam/conv2d_13/kernel/v training_8/Adam/conv2d_13/bias/v!training_8/Adam/dense_27/kernel/vtraining_8/Adam/dense_27/bias/v!training_8/Adam/dense_28/kernel/vtraining_8/Adam/dense_28/bias/v!training_8/Adam/dense_29/kernel/vtraining_8/Adam/dense_29/bias/v!training_8/Adam/dense_30/kernel/vtraining_8/Adam/dense_30/bias/v!training_8/Adam/dense_31/kernel/vtraining_8/Adam/dense_31/bias/v!training_8/Adam/dense_32/kernel/vtraining_8/Adam/dense_32/bias/v!training_8/Adam/dense_33/kernel/vtraining_8/Adam/dense_33/bias/v!training_8/Adam/dense_34/kernel/vtraining_8/Adam/dense_34/bias/v!training_8/Adam/dense_35/kernel/vtraining_8/Adam/dense_35/bias/v*e
Tin^
\2Z*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_12251??
?
?
)__inference_conv2d_13_layer_call_fn_11506

inputs
conv2d_13_kernel
conv2d_13_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_13_kernelconv2d_13_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_106662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
D__inference_conv2d_13_layer_call_and_return_conditional_losses_10666

inputs*
&conv2d_readvariableop_conv2d_13_kernel)
%biasadd_readvariableop_conv2d_13_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp&conv2d_readvariableop_conv2d_13_kernel*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_conv2d_13_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
D__inference_conv2d_11_layer_call_and_return_conditional_losses_11463

inputs*
&conv2d_readvariableop_conv2d_11_kernel)
%biasadd_readvariableop_conv2d_11_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp&conv2d_readvariableop_conv2d_11_kernel*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_conv2d_11_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
C__inference_dense_32_layer_call_and_return_conditional_losses_10847

inputs)
%matmul_readvariableop_dense_32_kernel(
$biasadd_readvariableop_dense_32_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_32_kernel*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_32_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
C__inference_dense_30_layer_call_and_return_conditional_losses_10893

inputs)
%matmul_readvariableop_dense_30_kernel(
$biasadd_readvariableop_dense_30_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_30_kernel*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_30_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
C__inference_dense_35_layer_call_and_return_conditional_losses_11672

inputs)
%matmul_readvariableop_dense_35_kernel(
$biasadd_readvariableop_dense_35_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_35_kernel*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_35_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_10_layer_call_fn_10559

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_105562
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
??
?(
__inference__traced_save_11974
file_prefix/
+savev2_conv2d_11_kernel_read_readvariableop-
)savev2_conv2d_11_bias_read_readvariableop/
+savev2_conv2d_12_kernel_read_readvariableop-
)savev2_conv2d_12_bias_read_readvariableop/
+savev2_conv2d_13_kernel_read_readvariableop-
)savev2_conv2d_13_bias_read_readvariableop.
*savev2_dense_27_kernel_read_readvariableop,
(savev2_dense_27_bias_read_readvariableop.
*savev2_dense_28_kernel_read_readvariableop,
(savev2_dense_28_bias_read_readvariableop.
*savev2_dense_29_kernel_read_readvariableop,
(savev2_dense_29_bias_read_readvariableop.
*savev2_dense_30_kernel_read_readvariableop,
(savev2_dense_30_bias_read_readvariableop.
*savev2_dense_31_kernel_read_readvariableop,
(savev2_dense_31_bias_read_readvariableop.
*savev2_dense_32_kernel_read_readvariableop,
(savev2_dense_32_bias_read_readvariableop.
*savev2_dense_33_kernel_read_readvariableop,
(savev2_dense_33_bias_read_readvariableop.
*savev2_dense_34_kernel_read_readvariableop,
(savev2_dense_34_bias_read_readvariableop.
*savev2_dense_35_kernel_read_readvariableop,
(savev2_dense_35_bias_read_readvariableop3
/savev2_training_8_adam_iter_read_readvariableop	5
1savev2_training_8_adam_beta_1_read_readvariableop5
1savev2_training_8_adam_beta_2_read_readvariableop4
0savev2_training_8_adam_decay_read_readvariableop<
8savev2_training_8_adam_learning_rate_read_readvariableop'
#savev2_total_48_read_readvariableop'
#savev2_count_48_read_readvariableop'
#savev2_total_49_read_readvariableop'
#savev2_count_49_read_readvariableop'
#savev2_total_50_read_readvariableop'
#savev2_count_50_read_readvariableop'
#savev2_total_51_read_readvariableop'
#savev2_count_51_read_readvariableop'
#savev2_total_52_read_readvariableop'
#savev2_count_52_read_readvariableop'
#savev2_total_53_read_readvariableop'
#savev2_count_53_read_readvariableopA
=savev2_training_8_adam_conv2d_11_kernel_m_read_readvariableop?
;savev2_training_8_adam_conv2d_11_bias_m_read_readvariableopA
=savev2_training_8_adam_conv2d_12_kernel_m_read_readvariableop?
;savev2_training_8_adam_conv2d_12_bias_m_read_readvariableopA
=savev2_training_8_adam_conv2d_13_kernel_m_read_readvariableop?
;savev2_training_8_adam_conv2d_13_bias_m_read_readvariableop@
<savev2_training_8_adam_dense_27_kernel_m_read_readvariableop>
:savev2_training_8_adam_dense_27_bias_m_read_readvariableop@
<savev2_training_8_adam_dense_28_kernel_m_read_readvariableop>
:savev2_training_8_adam_dense_28_bias_m_read_readvariableop@
<savev2_training_8_adam_dense_29_kernel_m_read_readvariableop>
:savev2_training_8_adam_dense_29_bias_m_read_readvariableop@
<savev2_training_8_adam_dense_30_kernel_m_read_readvariableop>
:savev2_training_8_adam_dense_30_bias_m_read_readvariableop@
<savev2_training_8_adam_dense_31_kernel_m_read_readvariableop>
:savev2_training_8_adam_dense_31_bias_m_read_readvariableop@
<savev2_training_8_adam_dense_32_kernel_m_read_readvariableop>
:savev2_training_8_adam_dense_32_bias_m_read_readvariableop@
<savev2_training_8_adam_dense_33_kernel_m_read_readvariableop>
:savev2_training_8_adam_dense_33_bias_m_read_readvariableop@
<savev2_training_8_adam_dense_34_kernel_m_read_readvariableop>
:savev2_training_8_adam_dense_34_bias_m_read_readvariableop@
<savev2_training_8_adam_dense_35_kernel_m_read_readvariableop>
:savev2_training_8_adam_dense_35_bias_m_read_readvariableopA
=savev2_training_8_adam_conv2d_11_kernel_v_read_readvariableop?
;savev2_training_8_adam_conv2d_11_bias_v_read_readvariableopA
=savev2_training_8_adam_conv2d_12_kernel_v_read_readvariableop?
;savev2_training_8_adam_conv2d_12_bias_v_read_readvariableopA
=savev2_training_8_adam_conv2d_13_kernel_v_read_readvariableop?
;savev2_training_8_adam_conv2d_13_bias_v_read_readvariableop@
<savev2_training_8_adam_dense_27_kernel_v_read_readvariableop>
:savev2_training_8_adam_dense_27_bias_v_read_readvariableop@
<savev2_training_8_adam_dense_28_kernel_v_read_readvariableop>
:savev2_training_8_adam_dense_28_bias_v_read_readvariableop@
<savev2_training_8_adam_dense_29_kernel_v_read_readvariableop>
:savev2_training_8_adam_dense_29_bias_v_read_readvariableop@
<savev2_training_8_adam_dense_30_kernel_v_read_readvariableop>
:savev2_training_8_adam_dense_30_bias_v_read_readvariableop@
<savev2_training_8_adam_dense_31_kernel_v_read_readvariableop>
:savev2_training_8_adam_dense_31_bias_v_read_readvariableop@
<savev2_training_8_adam_dense_32_kernel_v_read_readvariableop>
:savev2_training_8_adam_dense_32_bias_v_read_readvariableop@
<savev2_training_8_adam_dense_33_kernel_v_read_readvariableop>
:savev2_training_8_adam_dense_33_bias_v_read_readvariableop@
<savev2_training_8_adam_dense_34_kernel_v_read_readvariableop>
:savev2_training_8_adam_dense_34_bias_v_read_readvariableop@
<savev2_training_8_adam_dense_35_kernel_v_read_readvariableop>
:savev2_training_8_adam_dense_35_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?1
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Z*
dtype0*?0
value?0B?0ZB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Z*
dtype0*?
value?B?ZB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?&
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_11_kernel_read_readvariableop)savev2_conv2d_11_bias_read_readvariableop+savev2_conv2d_12_kernel_read_readvariableop)savev2_conv2d_12_bias_read_readvariableop+savev2_conv2d_13_kernel_read_readvariableop)savev2_conv2d_13_bias_read_readvariableop*savev2_dense_27_kernel_read_readvariableop(savev2_dense_27_bias_read_readvariableop*savev2_dense_28_kernel_read_readvariableop(savev2_dense_28_bias_read_readvariableop*savev2_dense_29_kernel_read_readvariableop(savev2_dense_29_bias_read_readvariableop*savev2_dense_30_kernel_read_readvariableop(savev2_dense_30_bias_read_readvariableop*savev2_dense_31_kernel_read_readvariableop(savev2_dense_31_bias_read_readvariableop*savev2_dense_32_kernel_read_readvariableop(savev2_dense_32_bias_read_readvariableop*savev2_dense_33_kernel_read_readvariableop(savev2_dense_33_bias_read_readvariableop*savev2_dense_34_kernel_read_readvariableop(savev2_dense_34_bias_read_readvariableop*savev2_dense_35_kernel_read_readvariableop(savev2_dense_35_bias_read_readvariableop/savev2_training_8_adam_iter_read_readvariableop1savev2_training_8_adam_beta_1_read_readvariableop1savev2_training_8_adam_beta_2_read_readvariableop0savev2_training_8_adam_decay_read_readvariableop8savev2_training_8_adam_learning_rate_read_readvariableop#savev2_total_48_read_readvariableop#savev2_count_48_read_readvariableop#savev2_total_49_read_readvariableop#savev2_count_49_read_readvariableop#savev2_total_50_read_readvariableop#savev2_count_50_read_readvariableop#savev2_total_51_read_readvariableop#savev2_count_51_read_readvariableop#savev2_total_52_read_readvariableop#savev2_count_52_read_readvariableop#savev2_total_53_read_readvariableop#savev2_count_53_read_readvariableop=savev2_training_8_adam_conv2d_11_kernel_m_read_readvariableop;savev2_training_8_adam_conv2d_11_bias_m_read_readvariableop=savev2_training_8_adam_conv2d_12_kernel_m_read_readvariableop;savev2_training_8_adam_conv2d_12_bias_m_read_readvariableop=savev2_training_8_adam_conv2d_13_kernel_m_read_readvariableop;savev2_training_8_adam_conv2d_13_bias_m_read_readvariableop<savev2_training_8_adam_dense_27_kernel_m_read_readvariableop:savev2_training_8_adam_dense_27_bias_m_read_readvariableop<savev2_training_8_adam_dense_28_kernel_m_read_readvariableop:savev2_training_8_adam_dense_28_bias_m_read_readvariableop<savev2_training_8_adam_dense_29_kernel_m_read_readvariableop:savev2_training_8_adam_dense_29_bias_m_read_readvariableop<savev2_training_8_adam_dense_30_kernel_m_read_readvariableop:savev2_training_8_adam_dense_30_bias_m_read_readvariableop<savev2_training_8_adam_dense_31_kernel_m_read_readvariableop:savev2_training_8_adam_dense_31_bias_m_read_readvariableop<savev2_training_8_adam_dense_32_kernel_m_read_readvariableop:savev2_training_8_adam_dense_32_bias_m_read_readvariableop<savev2_training_8_adam_dense_33_kernel_m_read_readvariableop:savev2_training_8_adam_dense_33_bias_m_read_readvariableop<savev2_training_8_adam_dense_34_kernel_m_read_readvariableop:savev2_training_8_adam_dense_34_bias_m_read_readvariableop<savev2_training_8_adam_dense_35_kernel_m_read_readvariableop:savev2_training_8_adam_dense_35_bias_m_read_readvariableop=savev2_training_8_adam_conv2d_11_kernel_v_read_readvariableop;savev2_training_8_adam_conv2d_11_bias_v_read_readvariableop=savev2_training_8_adam_conv2d_12_kernel_v_read_readvariableop;savev2_training_8_adam_conv2d_12_bias_v_read_readvariableop=savev2_training_8_adam_conv2d_13_kernel_v_read_readvariableop;savev2_training_8_adam_conv2d_13_bias_v_read_readvariableop<savev2_training_8_adam_dense_27_kernel_v_read_readvariableop:savev2_training_8_adam_dense_27_bias_v_read_readvariableop<savev2_training_8_adam_dense_28_kernel_v_read_readvariableop:savev2_training_8_adam_dense_28_bias_v_read_readvariableop<savev2_training_8_adam_dense_29_kernel_v_read_readvariableop:savev2_training_8_adam_dense_29_bias_v_read_readvariableop<savev2_training_8_adam_dense_30_kernel_v_read_readvariableop:savev2_training_8_adam_dense_30_bias_v_read_readvariableop<savev2_training_8_adam_dense_31_kernel_v_read_readvariableop:savev2_training_8_adam_dense_31_bias_v_read_readvariableop<savev2_training_8_adam_dense_32_kernel_v_read_readvariableop:savev2_training_8_adam_dense_32_bias_v_read_readvariableop<savev2_training_8_adam_dense_33_kernel_v_read_readvariableop:savev2_training_8_adam_dense_33_bias_v_read_readvariableop<savev2_training_8_adam_dense_34_kernel_v_read_readvariableop:savev2_training_8_adam_dense_34_bias_v_read_readvariableop<savev2_training_8_adam_dense_35_kernel_v_read_readvariableop:savev2_training_8_adam_dense_35_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *h
dtypes^
\2Z	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::: : : @:@:
??:?:	?@:@:@ : : :: :: :: :: :: :: : : : : : : : : : : : : : : : : ::: : : @:@:
??:?:	?@:@:@ : : :: :: :: :: :: :::: : : @:@:
??:?:	?@:@:@ : : :: :: :: :: :: :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%	!

_output_shapes
:	?@: 


_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :,*(
&
_output_shapes
:: +

_output_shapes
::,,(
&
_output_shapes
: : -

_output_shapes
: :,.(
&
_output_shapes
: @: /

_output_shapes
:@:&0"
 
_output_shapes
:
??:!1

_output_shapes	
:?:%2!

_output_shapes
:	?@: 3

_output_shapes
:@:$4 

_output_shapes

:@ : 5

_output_shapes
: :$6 

_output_shapes

: : 7

_output_shapes
::$8 

_output_shapes

: : 9

_output_shapes
::$: 

_output_shapes

: : ;

_output_shapes
::$< 

_output_shapes

: : =

_output_shapes
::$> 

_output_shapes

: : ?

_output_shapes
::$@ 

_output_shapes

: : A

_output_shapes
::,B(
&
_output_shapes
:: C

_output_shapes
::,D(
&
_output_shapes
: : E

_output_shapes
: :,F(
&
_output_shapes
: @: G

_output_shapes
:@:&H"
 
_output_shapes
:
??:!I

_output_shapes	
:?:%J!

_output_shapes
:	?@: K

_output_shapes
:@:$L 

_output_shapes

:@ : M

_output_shapes
: :$N 

_output_shapes

: : O

_output_shapes
::$P 

_output_shapes

: : Q

_output_shapes
::$R 

_output_shapes

: : S

_output_shapes
::$T 

_output_shapes

: : U

_output_shapes
::$V 

_output_shapes

: : W

_output_shapes
::$X 

_output_shapes

: : Y

_output_shapes
::Z

_output_shapes
: 
?

?
C__inference_dense_31_layer_call_and_return_conditional_losses_11600

inputs)
%matmul_readvariableop_dense_31_kernel(
$biasadd_readvariableop_dense_31_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_31_kernel*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_31_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
ľ
?
 __inference__wrapped_model_10542
input_4<
8model_3_conv2d_11_conv2d_readvariableop_conv2d_11_kernel;
7model_3_conv2d_11_biasadd_readvariableop_conv2d_11_bias<
8model_3_conv2d_12_conv2d_readvariableop_conv2d_12_kernel;
7model_3_conv2d_12_biasadd_readvariableop_conv2d_12_bias<
8model_3_conv2d_13_conv2d_readvariableop_conv2d_13_kernel;
7model_3_conv2d_13_biasadd_readvariableop_conv2d_13_bias:
6model_3_dense_27_matmul_readvariableop_dense_27_kernel9
5model_3_dense_27_biasadd_readvariableop_dense_27_bias:
6model_3_dense_28_matmul_readvariableop_dense_28_kernel9
5model_3_dense_28_biasadd_readvariableop_dense_28_bias:
6model_3_dense_29_matmul_readvariableop_dense_29_kernel9
5model_3_dense_29_biasadd_readvariableop_dense_29_bias:
6model_3_dense_35_matmul_readvariableop_dense_35_kernel9
5model_3_dense_35_biasadd_readvariableop_dense_35_bias:
6model_3_dense_34_matmul_readvariableop_dense_34_kernel9
5model_3_dense_34_biasadd_readvariableop_dense_34_bias:
6model_3_dense_33_matmul_readvariableop_dense_33_kernel9
5model_3_dense_33_biasadd_readvariableop_dense_33_bias:
6model_3_dense_32_matmul_readvariableop_dense_32_kernel9
5model_3_dense_32_biasadd_readvariableop_dense_32_bias:
6model_3_dense_31_matmul_readvariableop_dense_31_kernel9
5model_3_dense_31_biasadd_readvariableop_dense_31_bias:
6model_3_dense_30_matmul_readvariableop_dense_30_kernel9
5model_3_dense_30_biasadd_readvariableop_dense_30_bias
identity

identity_1

identity_2

identity_3

identity_4

identity_5??(model_3/conv2d_11/BiasAdd/ReadVariableOp?'model_3/conv2d_11/Conv2D/ReadVariableOp?(model_3/conv2d_12/BiasAdd/ReadVariableOp?'model_3/conv2d_12/Conv2D/ReadVariableOp?(model_3/conv2d_13/BiasAdd/ReadVariableOp?'model_3/conv2d_13/Conv2D/ReadVariableOp?'model_3/dense_27/BiasAdd/ReadVariableOp?&model_3/dense_27/MatMul/ReadVariableOp?'model_3/dense_28/BiasAdd/ReadVariableOp?&model_3/dense_28/MatMul/ReadVariableOp?'model_3/dense_29/BiasAdd/ReadVariableOp?&model_3/dense_29/MatMul/ReadVariableOp?'model_3/dense_30/BiasAdd/ReadVariableOp?&model_3/dense_30/MatMul/ReadVariableOp?'model_3/dense_31/BiasAdd/ReadVariableOp?&model_3/dense_31/MatMul/ReadVariableOp?'model_3/dense_32/BiasAdd/ReadVariableOp?&model_3/dense_32/MatMul/ReadVariableOp?'model_3/dense_33/BiasAdd/ReadVariableOp?&model_3/dense_33/MatMul/ReadVariableOp?'model_3/dense_34/BiasAdd/ReadVariableOp?&model_3/dense_34/MatMul/ReadVariableOp?'model_3/dense_35/BiasAdd/ReadVariableOp?&model_3/dense_35/MatMul/ReadVariableOp?
'model_3/conv2d_11/Conv2D/ReadVariableOpReadVariableOp8model_3_conv2d_11_conv2d_readvariableop_conv2d_11_kernel*&
_output_shapes
:*
dtype02)
'model_3/conv2d_11/Conv2D/ReadVariableOp?
model_3/conv2d_11/Conv2DConv2Dinput_4/model_3/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
model_3/conv2d_11/Conv2D?
(model_3/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp7model_3_conv2d_11_biasadd_readvariableop_conv2d_11_bias*
_output_shapes
:*
dtype02*
(model_3/conv2d_11/BiasAdd/ReadVariableOp?
model_3/conv2d_11/BiasAddBiasAdd!model_3/conv2d_11/Conv2D:output:00model_3/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
model_3/conv2d_11/BiasAdd?
model_3/conv2d_11/ReluRelu"model_3/conv2d_11/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
model_3/conv2d_11/Relu?
 model_3/max_pooling2d_10/MaxPoolMaxPool$model_3/conv2d_11/Relu:activations:0*/
_output_shapes
:?????????***
ksize
*
paddingVALID*
strides
2"
 model_3/max_pooling2d_10/MaxPool?
'model_3/conv2d_12/Conv2D/ReadVariableOpReadVariableOp8model_3_conv2d_12_conv2d_readvariableop_conv2d_12_kernel*&
_output_shapes
: *
dtype02)
'model_3/conv2d_12/Conv2D/ReadVariableOp?
model_3/conv2d_12/Conv2DConv2D)model_3/max_pooling2d_10/MaxPool:output:0/model_3/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(( *
paddingVALID*
strides
2
model_3/conv2d_12/Conv2D?
(model_3/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp7model_3_conv2d_12_biasadd_readvariableop_conv2d_12_bias*
_output_shapes
: *
dtype02*
(model_3/conv2d_12/BiasAdd/ReadVariableOp?
model_3/conv2d_12/BiasAddBiasAdd!model_3/conv2d_12/Conv2D:output:00model_3/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(( 2
model_3/conv2d_12/BiasAdd?
model_3/conv2d_12/ReluRelu"model_3/conv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(( 2
model_3/conv2d_12/Relu?
 model_3/max_pooling2d_11/MaxPoolMaxPool$model_3/conv2d_12/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2"
 model_3/max_pooling2d_11/MaxPool?
'model_3/conv2d_13/Conv2D/ReadVariableOpReadVariableOp8model_3_conv2d_13_conv2d_readvariableop_conv2d_13_kernel*&
_output_shapes
: @*
dtype02)
'model_3/conv2d_13/Conv2D/ReadVariableOp?
model_3/conv2d_13/Conv2DConv2D)model_3/max_pooling2d_11/MaxPool:output:0/model_3/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
model_3/conv2d_13/Conv2D?
(model_3/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp7model_3_conv2d_13_biasadd_readvariableop_conv2d_13_bias*
_output_shapes
:@*
dtype02*
(model_3/conv2d_13/BiasAdd/ReadVariableOp?
model_3/conv2d_13/BiasAddBiasAdd!model_3/conv2d_13/Conv2D:output:00model_3/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
model_3/conv2d_13/BiasAdd?
model_3/conv2d_13/ReluRelu"model_3/conv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
model_3/conv2d_13/Relu?
 model_3/max_pooling2d_12/MaxPoolMaxPool$model_3/conv2d_13/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2"
 model_3/max_pooling2d_12/MaxPool?
model_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  2
model_3/flatten_3/Const?
model_3/flatten_3/ReshapeReshape)model_3/max_pooling2d_12/MaxPool:output:0 model_3/flatten_3/Const:output:0*
T0*(
_output_shapes
:??????????2
model_3/flatten_3/Reshape?
&model_3/dense_27/MatMul/ReadVariableOpReadVariableOp6model_3_dense_27_matmul_readvariableop_dense_27_kernel* 
_output_shapes
:
??*
dtype02(
&model_3/dense_27/MatMul/ReadVariableOp?
model_3/dense_27/MatMulMatMul"model_3/flatten_3/Reshape:output:0.model_3/dense_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_3/dense_27/MatMul?
'model_3/dense_27/BiasAdd/ReadVariableOpReadVariableOp5model_3_dense_27_biasadd_readvariableop_dense_27_bias*
_output_shapes	
:?*
dtype02)
'model_3/dense_27/BiasAdd/ReadVariableOp?
model_3/dense_27/BiasAddBiasAdd!model_3/dense_27/MatMul:product:0/model_3/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_3/dense_27/BiasAdd?
model_3/dense_27/ReluRelu!model_3/dense_27/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_3/dense_27/Relu?
&model_3/dense_28/MatMul/ReadVariableOpReadVariableOp6model_3_dense_28_matmul_readvariableop_dense_28_kernel*
_output_shapes
:	?@*
dtype02(
&model_3/dense_28/MatMul/ReadVariableOp?
model_3/dense_28/MatMulMatMul#model_3/dense_27/Relu:activations:0.model_3/dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_3/dense_28/MatMul?
'model_3/dense_28/BiasAdd/ReadVariableOpReadVariableOp5model_3_dense_28_biasadd_readvariableop_dense_28_bias*
_output_shapes
:@*
dtype02)
'model_3/dense_28/BiasAdd/ReadVariableOp?
model_3/dense_28/BiasAddBiasAdd!model_3/dense_28/MatMul:product:0/model_3/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_3/dense_28/BiasAdd?
model_3/dense_28/ReluRelu!model_3/dense_28/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model_3/dense_28/Relu?
&model_3/dense_29/MatMul/ReadVariableOpReadVariableOp6model_3_dense_29_matmul_readvariableop_dense_29_kernel*
_output_shapes

:@ *
dtype02(
&model_3/dense_29/MatMul/ReadVariableOp?
model_3/dense_29/MatMulMatMul#model_3/dense_28/Relu:activations:0.model_3/dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
model_3/dense_29/MatMul?
'model_3/dense_29/BiasAdd/ReadVariableOpReadVariableOp5model_3_dense_29_biasadd_readvariableop_dense_29_bias*
_output_shapes
: *
dtype02)
'model_3/dense_29/BiasAdd/ReadVariableOp?
model_3/dense_29/BiasAddBiasAdd!model_3/dense_29/MatMul:product:0/model_3/dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
model_3/dense_29/BiasAdd?
model_3/dense_29/ReluRelu!model_3/dense_29/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
model_3/dense_29/Relu?
&model_3/dense_35/MatMul/ReadVariableOpReadVariableOp6model_3_dense_35_matmul_readvariableop_dense_35_kernel*
_output_shapes

: *
dtype02(
&model_3/dense_35/MatMul/ReadVariableOp?
model_3/dense_35/MatMulMatMul#model_3/dense_29/Relu:activations:0.model_3/dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_3/dense_35/MatMul?
'model_3/dense_35/BiasAdd/ReadVariableOpReadVariableOp5model_3_dense_35_biasadd_readvariableop_dense_35_bias*
_output_shapes
:*
dtype02)
'model_3/dense_35/BiasAdd/ReadVariableOp?
model_3/dense_35/BiasAddBiasAdd!model_3/dense_35/MatMul:product:0/model_3/dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_3/dense_35/BiasAdd?
model_3/dense_35/SigmoidSigmoid!model_3/dense_35/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_3/dense_35/Sigmoid?
&model_3/dense_34/MatMul/ReadVariableOpReadVariableOp6model_3_dense_34_matmul_readvariableop_dense_34_kernel*
_output_shapes

: *
dtype02(
&model_3/dense_34/MatMul/ReadVariableOp?
model_3/dense_34/MatMulMatMul#model_3/dense_29/Relu:activations:0.model_3/dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_3/dense_34/MatMul?
'model_3/dense_34/BiasAdd/ReadVariableOpReadVariableOp5model_3_dense_34_biasadd_readvariableop_dense_34_bias*
_output_shapes
:*
dtype02)
'model_3/dense_34/BiasAdd/ReadVariableOp?
model_3/dense_34/BiasAddBiasAdd!model_3/dense_34/MatMul:product:0/model_3/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_3/dense_34/BiasAdd?
model_3/dense_34/SigmoidSigmoid!model_3/dense_34/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_3/dense_34/Sigmoid?
&model_3/dense_33/MatMul/ReadVariableOpReadVariableOp6model_3_dense_33_matmul_readvariableop_dense_33_kernel*
_output_shapes

: *
dtype02(
&model_3/dense_33/MatMul/ReadVariableOp?
model_3/dense_33/MatMulMatMul#model_3/dense_29/Relu:activations:0.model_3/dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_3/dense_33/MatMul?
'model_3/dense_33/BiasAdd/ReadVariableOpReadVariableOp5model_3_dense_33_biasadd_readvariableop_dense_33_bias*
_output_shapes
:*
dtype02)
'model_3/dense_33/BiasAdd/ReadVariableOp?
model_3/dense_33/BiasAddBiasAdd!model_3/dense_33/MatMul:product:0/model_3/dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_3/dense_33/BiasAdd?
model_3/dense_33/SigmoidSigmoid!model_3/dense_33/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_3/dense_33/Sigmoid?
&model_3/dense_32/MatMul/ReadVariableOpReadVariableOp6model_3_dense_32_matmul_readvariableop_dense_32_kernel*
_output_shapes

: *
dtype02(
&model_3/dense_32/MatMul/ReadVariableOp?
model_3/dense_32/MatMulMatMul#model_3/dense_29/Relu:activations:0.model_3/dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_3/dense_32/MatMul?
'model_3/dense_32/BiasAdd/ReadVariableOpReadVariableOp5model_3_dense_32_biasadd_readvariableop_dense_32_bias*
_output_shapes
:*
dtype02)
'model_3/dense_32/BiasAdd/ReadVariableOp?
model_3/dense_32/BiasAddBiasAdd!model_3/dense_32/MatMul:product:0/model_3/dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_3/dense_32/BiasAdd?
model_3/dense_32/SigmoidSigmoid!model_3/dense_32/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_3/dense_32/Sigmoid?
&model_3/dense_31/MatMul/ReadVariableOpReadVariableOp6model_3_dense_31_matmul_readvariableop_dense_31_kernel*
_output_shapes

: *
dtype02(
&model_3/dense_31/MatMul/ReadVariableOp?
model_3/dense_31/MatMulMatMul#model_3/dense_29/Relu:activations:0.model_3/dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_3/dense_31/MatMul?
'model_3/dense_31/BiasAdd/ReadVariableOpReadVariableOp5model_3_dense_31_biasadd_readvariableop_dense_31_bias*
_output_shapes
:*
dtype02)
'model_3/dense_31/BiasAdd/ReadVariableOp?
model_3/dense_31/BiasAddBiasAdd!model_3/dense_31/MatMul:product:0/model_3/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_3/dense_31/BiasAdd?
model_3/dense_31/SigmoidSigmoid!model_3/dense_31/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_3/dense_31/Sigmoid?
&model_3/dense_30/MatMul/ReadVariableOpReadVariableOp6model_3_dense_30_matmul_readvariableop_dense_30_kernel*
_output_shapes

: *
dtype02(
&model_3/dense_30/MatMul/ReadVariableOp?
model_3/dense_30/MatMulMatMul#model_3/dense_29/Relu:activations:0.model_3/dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_3/dense_30/MatMul?
'model_3/dense_30/BiasAdd/ReadVariableOpReadVariableOp5model_3_dense_30_biasadd_readvariableop_dense_30_bias*
_output_shapes
:*
dtype02)
'model_3/dense_30/BiasAdd/ReadVariableOp?
model_3/dense_30/BiasAddBiasAdd!model_3/dense_30/MatMul:product:0/model_3/dense_30/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_3/dense_30/BiasAdd?
model_3/dense_30/SigmoidSigmoid!model_3/dense_30/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_3/dense_30/Sigmoid?
IdentityIdentitymodel_3/dense_30/Sigmoid:y:0)^model_3/conv2d_11/BiasAdd/ReadVariableOp(^model_3/conv2d_11/Conv2D/ReadVariableOp)^model_3/conv2d_12/BiasAdd/ReadVariableOp(^model_3/conv2d_12/Conv2D/ReadVariableOp)^model_3/conv2d_13/BiasAdd/ReadVariableOp(^model_3/conv2d_13/Conv2D/ReadVariableOp(^model_3/dense_27/BiasAdd/ReadVariableOp'^model_3/dense_27/MatMul/ReadVariableOp(^model_3/dense_28/BiasAdd/ReadVariableOp'^model_3/dense_28/MatMul/ReadVariableOp(^model_3/dense_29/BiasAdd/ReadVariableOp'^model_3/dense_29/MatMul/ReadVariableOp(^model_3/dense_30/BiasAdd/ReadVariableOp'^model_3/dense_30/MatMul/ReadVariableOp(^model_3/dense_31/BiasAdd/ReadVariableOp'^model_3/dense_31/MatMul/ReadVariableOp(^model_3/dense_32/BiasAdd/ReadVariableOp'^model_3/dense_32/MatMul/ReadVariableOp(^model_3/dense_33/BiasAdd/ReadVariableOp'^model_3/dense_33/MatMul/ReadVariableOp(^model_3/dense_34/BiasAdd/ReadVariableOp'^model_3/dense_34/MatMul/ReadVariableOp(^model_3/dense_35/BiasAdd/ReadVariableOp'^model_3/dense_35/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitymodel_3/dense_31/Sigmoid:y:0)^model_3/conv2d_11/BiasAdd/ReadVariableOp(^model_3/conv2d_11/Conv2D/ReadVariableOp)^model_3/conv2d_12/BiasAdd/ReadVariableOp(^model_3/conv2d_12/Conv2D/ReadVariableOp)^model_3/conv2d_13/BiasAdd/ReadVariableOp(^model_3/conv2d_13/Conv2D/ReadVariableOp(^model_3/dense_27/BiasAdd/ReadVariableOp'^model_3/dense_27/MatMul/ReadVariableOp(^model_3/dense_28/BiasAdd/ReadVariableOp'^model_3/dense_28/MatMul/ReadVariableOp(^model_3/dense_29/BiasAdd/ReadVariableOp'^model_3/dense_29/MatMul/ReadVariableOp(^model_3/dense_30/BiasAdd/ReadVariableOp'^model_3/dense_30/MatMul/ReadVariableOp(^model_3/dense_31/BiasAdd/ReadVariableOp'^model_3/dense_31/MatMul/ReadVariableOp(^model_3/dense_32/BiasAdd/ReadVariableOp'^model_3/dense_32/MatMul/ReadVariableOp(^model_3/dense_33/BiasAdd/ReadVariableOp'^model_3/dense_33/MatMul/ReadVariableOp(^model_3/dense_34/BiasAdd/ReadVariableOp'^model_3/dense_34/MatMul/ReadVariableOp(^model_3/dense_35/BiasAdd/ReadVariableOp'^model_3/dense_35/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identitymodel_3/dense_32/Sigmoid:y:0)^model_3/conv2d_11/BiasAdd/ReadVariableOp(^model_3/conv2d_11/Conv2D/ReadVariableOp)^model_3/conv2d_12/BiasAdd/ReadVariableOp(^model_3/conv2d_12/Conv2D/ReadVariableOp)^model_3/conv2d_13/BiasAdd/ReadVariableOp(^model_3/conv2d_13/Conv2D/ReadVariableOp(^model_3/dense_27/BiasAdd/ReadVariableOp'^model_3/dense_27/MatMul/ReadVariableOp(^model_3/dense_28/BiasAdd/ReadVariableOp'^model_3/dense_28/MatMul/ReadVariableOp(^model_3/dense_29/BiasAdd/ReadVariableOp'^model_3/dense_29/MatMul/ReadVariableOp(^model_3/dense_30/BiasAdd/ReadVariableOp'^model_3/dense_30/MatMul/ReadVariableOp(^model_3/dense_31/BiasAdd/ReadVariableOp'^model_3/dense_31/MatMul/ReadVariableOp(^model_3/dense_32/BiasAdd/ReadVariableOp'^model_3/dense_32/MatMul/ReadVariableOp(^model_3/dense_33/BiasAdd/ReadVariableOp'^model_3/dense_33/MatMul/ReadVariableOp(^model_3/dense_34/BiasAdd/ReadVariableOp'^model_3/dense_34/MatMul/ReadVariableOp(^model_3/dense_35/BiasAdd/ReadVariableOp'^model_3/dense_35/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identitymodel_3/dense_33/Sigmoid:y:0)^model_3/conv2d_11/BiasAdd/ReadVariableOp(^model_3/conv2d_11/Conv2D/ReadVariableOp)^model_3/conv2d_12/BiasAdd/ReadVariableOp(^model_3/conv2d_12/Conv2D/ReadVariableOp)^model_3/conv2d_13/BiasAdd/ReadVariableOp(^model_3/conv2d_13/Conv2D/ReadVariableOp(^model_3/dense_27/BiasAdd/ReadVariableOp'^model_3/dense_27/MatMul/ReadVariableOp(^model_3/dense_28/BiasAdd/ReadVariableOp'^model_3/dense_28/MatMul/ReadVariableOp(^model_3/dense_29/BiasAdd/ReadVariableOp'^model_3/dense_29/MatMul/ReadVariableOp(^model_3/dense_30/BiasAdd/ReadVariableOp'^model_3/dense_30/MatMul/ReadVariableOp(^model_3/dense_31/BiasAdd/ReadVariableOp'^model_3/dense_31/MatMul/ReadVariableOp(^model_3/dense_32/BiasAdd/ReadVariableOp'^model_3/dense_32/MatMul/ReadVariableOp(^model_3/dense_33/BiasAdd/ReadVariableOp'^model_3/dense_33/MatMul/ReadVariableOp(^model_3/dense_34/BiasAdd/ReadVariableOp'^model_3/dense_34/MatMul/ReadVariableOp(^model_3/dense_35/BiasAdd/ReadVariableOp'^model_3/dense_35/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identitymodel_3/dense_34/Sigmoid:y:0)^model_3/conv2d_11/BiasAdd/ReadVariableOp(^model_3/conv2d_11/Conv2D/ReadVariableOp)^model_3/conv2d_12/BiasAdd/ReadVariableOp(^model_3/conv2d_12/Conv2D/ReadVariableOp)^model_3/conv2d_13/BiasAdd/ReadVariableOp(^model_3/conv2d_13/Conv2D/ReadVariableOp(^model_3/dense_27/BiasAdd/ReadVariableOp'^model_3/dense_27/MatMul/ReadVariableOp(^model_3/dense_28/BiasAdd/ReadVariableOp'^model_3/dense_28/MatMul/ReadVariableOp(^model_3/dense_29/BiasAdd/ReadVariableOp'^model_3/dense_29/MatMul/ReadVariableOp(^model_3/dense_30/BiasAdd/ReadVariableOp'^model_3/dense_30/MatMul/ReadVariableOp(^model_3/dense_31/BiasAdd/ReadVariableOp'^model_3/dense_31/MatMul/ReadVariableOp(^model_3/dense_32/BiasAdd/ReadVariableOp'^model_3/dense_32/MatMul/ReadVariableOp(^model_3/dense_33/BiasAdd/ReadVariableOp'^model_3/dense_33/MatMul/ReadVariableOp(^model_3/dense_34/BiasAdd/ReadVariableOp'^model_3/dense_34/MatMul/ReadVariableOp(^model_3/dense_35/BiasAdd/ReadVariableOp'^model_3/dense_35/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identitymodel_3/dense_35/Sigmoid:y:0)^model_3/conv2d_11/BiasAdd/ReadVariableOp(^model_3/conv2d_11/Conv2D/ReadVariableOp)^model_3/conv2d_12/BiasAdd/ReadVariableOp(^model_3/conv2d_12/Conv2D/ReadVariableOp)^model_3/conv2d_13/BiasAdd/ReadVariableOp(^model_3/conv2d_13/Conv2D/ReadVariableOp(^model_3/dense_27/BiasAdd/ReadVariableOp'^model_3/dense_27/MatMul/ReadVariableOp(^model_3/dense_28/BiasAdd/ReadVariableOp'^model_3/dense_28/MatMul/ReadVariableOp(^model_3/dense_29/BiasAdd/ReadVariableOp'^model_3/dense_29/MatMul/ReadVariableOp(^model_3/dense_30/BiasAdd/ReadVariableOp'^model_3/dense_30/MatMul/ReadVariableOp(^model_3/dense_31/BiasAdd/ReadVariableOp'^model_3/dense_31/MatMul/ReadVariableOp(^model_3/dense_32/BiasAdd/ReadVariableOp'^model_3/dense_32/MatMul/ReadVariableOp(^model_3/dense_33/BiasAdd/ReadVariableOp'^model_3/dense_33/MatMul/ReadVariableOp(^model_3/dense_34/BiasAdd/ReadVariableOp'^model_3/dense_34/MatMul/ReadVariableOp(^model_3/dense_35/BiasAdd/ReadVariableOp'^model_3/dense_35/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*?
_input_shapes
}:???????????::::::::::::::::::::::::2T
(model_3/conv2d_11/BiasAdd/ReadVariableOp(model_3/conv2d_11/BiasAdd/ReadVariableOp2R
'model_3/conv2d_11/Conv2D/ReadVariableOp'model_3/conv2d_11/Conv2D/ReadVariableOp2T
(model_3/conv2d_12/BiasAdd/ReadVariableOp(model_3/conv2d_12/BiasAdd/ReadVariableOp2R
'model_3/conv2d_12/Conv2D/ReadVariableOp'model_3/conv2d_12/Conv2D/ReadVariableOp2T
(model_3/conv2d_13/BiasAdd/ReadVariableOp(model_3/conv2d_13/BiasAdd/ReadVariableOp2R
'model_3/conv2d_13/Conv2D/ReadVariableOp'model_3/conv2d_13/Conv2D/ReadVariableOp2R
'model_3/dense_27/BiasAdd/ReadVariableOp'model_3/dense_27/BiasAdd/ReadVariableOp2P
&model_3/dense_27/MatMul/ReadVariableOp&model_3/dense_27/MatMul/ReadVariableOp2R
'model_3/dense_28/BiasAdd/ReadVariableOp'model_3/dense_28/BiasAdd/ReadVariableOp2P
&model_3/dense_28/MatMul/ReadVariableOp&model_3/dense_28/MatMul/ReadVariableOp2R
'model_3/dense_29/BiasAdd/ReadVariableOp'model_3/dense_29/BiasAdd/ReadVariableOp2P
&model_3/dense_29/MatMul/ReadVariableOp&model_3/dense_29/MatMul/ReadVariableOp2R
'model_3/dense_30/BiasAdd/ReadVariableOp'model_3/dense_30/BiasAdd/ReadVariableOp2P
&model_3/dense_30/MatMul/ReadVariableOp&model_3/dense_30/MatMul/ReadVariableOp2R
'model_3/dense_31/BiasAdd/ReadVariableOp'model_3/dense_31/BiasAdd/ReadVariableOp2P
&model_3/dense_31/MatMul/ReadVariableOp&model_3/dense_31/MatMul/ReadVariableOp2R
'model_3/dense_32/BiasAdd/ReadVariableOp'model_3/dense_32/BiasAdd/ReadVariableOp2P
&model_3/dense_32/MatMul/ReadVariableOp&model_3/dense_32/MatMul/ReadVariableOp2R
'model_3/dense_33/BiasAdd/ReadVariableOp'model_3/dense_33/BiasAdd/ReadVariableOp2P
&model_3/dense_33/MatMul/ReadVariableOp&model_3/dense_33/MatMul/ReadVariableOp2R
'model_3/dense_34/BiasAdd/ReadVariableOp'model_3/dense_34/BiasAdd/ReadVariableOp2P
&model_3/dense_34/MatMul/ReadVariableOp&model_3/dense_34/MatMul/ReadVariableOp2R
'model_3/dense_35/BiasAdd/ReadVariableOp'model_3/dense_35/BiasAdd/ReadVariableOp2P
&model_3/dense_35/MatMul/ReadVariableOp&model_3/dense_35/MatMul/ReadVariableOp:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_4
?
g
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_10565

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
C__inference_dense_35_layer_call_and_return_conditional_losses_10778

inputs)
%matmul_readvariableop_dense_35_kernel(
$biasadd_readvariableop_dense_35_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_35_kernel*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_35_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?e
?

B__inference_model_3_layer_call_and_return_conditional_losses_10911
input_4
conv2d_11_conv2d_11_kernel
conv2d_11_conv2d_11_bias
conv2d_12_conv2d_12_kernel
conv2d_12_conv2d_12_bias
conv2d_13_conv2d_13_kernel
conv2d_13_conv2d_13_bias
dense_27_dense_27_kernel
dense_27_dense_27_bias
dense_28_dense_28_kernel
dense_28_dense_28_bias
dense_29_dense_29_kernel
dense_29_dense_29_bias
dense_35_dense_35_kernel
dense_35_dense_35_bias
dense_34_dense_34_kernel
dense_34_dense_34_bias
dense_33_dense_33_kernel
dense_33_dense_33_bias
dense_32_dense_32_kernel
dense_32_dense_32_bias
dense_31_dense_31_kernel
dense_31_dense_31_bias
dense_30_dense_30_kernel
dense_30_dense_30_bias
identity

identity_1

identity_2

identity_3

identity_4

identity_5??!conv2d_11/StatefulPartitionedCall?!conv2d_12/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall? dense_27/StatefulPartitionedCall? dense_28/StatefulPartitionedCall? dense_29/StatefulPartitionedCall? dense_30/StatefulPartitionedCall? dense_31/StatefulPartitionedCall? dense_32/StatefulPartitionedCall? dense_33/StatefulPartitionedCall? dense_34/StatefulPartitionedCall? dense_35/StatefulPartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCallinput_4conv2d_11_conv2d_11_kernelconv2d_11_conv2d_11_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_106082#
!conv2d_11/StatefulPartitionedCall?
 max_pooling2d_10/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_105562"
 max_pooling2d_10/PartitionedCall?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_10/PartitionedCall:output:0conv2d_12_conv2d_12_kernelconv2d_12_conv2d_12_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(( *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_106372#
!conv2d_12/StatefulPartitionedCall?
 max_pooling2d_11/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_105732"
 max_pooling2d_11/PartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_11/PartitionedCall:output:0conv2d_13_conv2d_13_kernelconv2d_13_conv2d_13_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_106662#
!conv2d_13/StatefulPartitionedCall?
 max_pooling2d_12/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_105902"
 max_pooling2d_12/PartitionedCall?
flatten_3/PartitionedCallPartitionedCall)max_pooling2d_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_106902
flatten_3/PartitionedCall?
 dense_27/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_27_dense_27_kerneldense_27_dense_27_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_27_layer_call_and_return_conditional_losses_107092"
 dense_27/StatefulPartitionedCall?
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0dense_28_dense_28_kerneldense_28_dense_28_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_28_layer_call_and_return_conditional_losses_107322"
 dense_28/StatefulPartitionedCall?
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_dense_29_kerneldense_29_dense_29_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_29_layer_call_and_return_conditional_losses_107552"
 dense_29/StatefulPartitionedCall?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_29/StatefulPartitionedCall:output:0dense_35_dense_35_kerneldense_35_dense_35_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_35_layer_call_and_return_conditional_losses_107782"
 dense_35/StatefulPartitionedCall?
 dense_34/StatefulPartitionedCallStatefulPartitionedCall)dense_29/StatefulPartitionedCall:output:0dense_34_dense_34_kerneldense_34_dense_34_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_34_layer_call_and_return_conditional_losses_108012"
 dense_34/StatefulPartitionedCall?
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_29/StatefulPartitionedCall:output:0dense_33_dense_33_kerneldense_33_dense_33_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_33_layer_call_and_return_conditional_losses_108242"
 dense_33/StatefulPartitionedCall?
 dense_32/StatefulPartitionedCallStatefulPartitionedCall)dense_29/StatefulPartitionedCall:output:0dense_32_dense_32_kerneldense_32_dense_32_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_32_layer_call_and_return_conditional_losses_108472"
 dense_32/StatefulPartitionedCall?
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_29/StatefulPartitionedCall:output:0dense_31_dense_31_kerneldense_31_dense_31_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_31_layer_call_and_return_conditional_losses_108702"
 dense_31/StatefulPartitionedCall?
 dense_30/StatefulPartitionedCallStatefulPartitionedCall)dense_29/StatefulPartitionedCall:output:0dense_30_dense_30_kerneldense_30_dense_30_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_30_layer_call_and_return_conditional_losses_108932"
 dense_30/StatefulPartitionedCall?
IdentityIdentity)dense_30/StatefulPartitionedCall:output:0"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity)dense_31/StatefulPartitionedCall:output:0"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity)dense_32/StatefulPartitionedCall:output:0"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity)dense_33/StatefulPartitionedCall:output:0"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity)dense_34/StatefulPartitionedCall:output:0"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity)dense_35/StatefulPartitionedCall:output:0"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*?
_input_shapes
}:???????????::::::::::::::::::::::::2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_4
?

?
C__inference_dense_28_layer_call_and_return_conditional_losses_10732

inputs)
%matmul_readvariableop_dense_28_kernel(
$biasadd_readvariableop_dense_28_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_28_kernel*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_28_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_dense_27_layer_call_and_return_conditional_losses_11528

inputs)
%matmul_readvariableop_dense_27_kernel(
$biasadd_readvariableop_dense_27_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_27_kernel* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_27_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_model_3_layer_call_fn_11413

inputs
conv2d_11_kernel
conv2d_11_bias
conv2d_12_kernel
conv2d_12_bias
conv2d_13_kernel
conv2d_13_bias
dense_27_kernel
dense_27_bias
dense_28_kernel
dense_28_bias
dense_29_kernel
dense_29_bias
dense_35_kernel
dense_35_bias
dense_34_kernel
dense_34_bias
dense_33_kernel
dense_33_bias
dense_32_kernel
dense_32_bias
dense_31_kernel
dense_31_bias
dense_30_kernel
dense_30_bias
identity

identity_1

identity_2

identity_3

identity_4

identity_5??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_11_kernelconv2d_11_biasconv2d_12_kernelconv2d_12_biasconv2d_13_kernelconv2d_13_biasdense_27_kerneldense_27_biasdense_28_kerneldense_28_biasdense_29_kerneldense_29_biasdense_35_kerneldense_35_biasdense_34_kerneldense_34_biasdense_33_kerneldense_33_biasdense_32_kerneldense_32_biasdense_31_kerneldense_31_biasdense_30_kerneldense_30_bias*$
Tin
2*
Tout

2*
_collective_manager_ids
 *?
_output_shapest
r:?????????:?????????:?????????:?????????:?????????:?????????*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_110122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*?
_input_shapes
}:???????????::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_3_layer_call_and_return_conditional_losses_10690

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_10582

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
ס
?
B__inference_model_3_layer_call_and_return_conditional_losses_11276

inputs4
0conv2d_11_conv2d_readvariableop_conv2d_11_kernel3
/conv2d_11_biasadd_readvariableop_conv2d_11_bias4
0conv2d_12_conv2d_readvariableop_conv2d_12_kernel3
/conv2d_12_biasadd_readvariableop_conv2d_12_bias4
0conv2d_13_conv2d_readvariableop_conv2d_13_kernel3
/conv2d_13_biasadd_readvariableop_conv2d_13_bias2
.dense_27_matmul_readvariableop_dense_27_kernel1
-dense_27_biasadd_readvariableop_dense_27_bias2
.dense_28_matmul_readvariableop_dense_28_kernel1
-dense_28_biasadd_readvariableop_dense_28_bias2
.dense_29_matmul_readvariableop_dense_29_kernel1
-dense_29_biasadd_readvariableop_dense_29_bias2
.dense_35_matmul_readvariableop_dense_35_kernel1
-dense_35_biasadd_readvariableop_dense_35_bias2
.dense_34_matmul_readvariableop_dense_34_kernel1
-dense_34_biasadd_readvariableop_dense_34_bias2
.dense_33_matmul_readvariableop_dense_33_kernel1
-dense_33_biasadd_readvariableop_dense_33_bias2
.dense_32_matmul_readvariableop_dense_32_kernel1
-dense_32_biasadd_readvariableop_dense_32_bias2
.dense_31_matmul_readvariableop_dense_31_kernel1
-dense_31_biasadd_readvariableop_dense_31_bias2
.dense_30_matmul_readvariableop_dense_30_kernel1
-dense_30_biasadd_readvariableop_dense_30_bias
identity

identity_1

identity_2

identity_3

identity_4

identity_5?? conv2d_11/BiasAdd/ReadVariableOp?conv2d_11/Conv2D/ReadVariableOp? conv2d_12/BiasAdd/ReadVariableOp?conv2d_12/Conv2D/ReadVariableOp? conv2d_13/BiasAdd/ReadVariableOp?conv2d_13/Conv2D/ReadVariableOp?dense_27/BiasAdd/ReadVariableOp?dense_27/MatMul/ReadVariableOp?dense_28/BiasAdd/ReadVariableOp?dense_28/MatMul/ReadVariableOp?dense_29/BiasAdd/ReadVariableOp?dense_29/MatMul/ReadVariableOp?dense_30/BiasAdd/ReadVariableOp?dense_30/MatMul/ReadVariableOp?dense_31/BiasAdd/ReadVariableOp?dense_31/MatMul/ReadVariableOp?dense_32/BiasAdd/ReadVariableOp?dense_32/MatMul/ReadVariableOp?dense_33/BiasAdd/ReadVariableOp?dense_33/MatMul/ReadVariableOp?dense_34/BiasAdd/ReadVariableOp?dense_34/MatMul/ReadVariableOp?dense_35/BiasAdd/ReadVariableOp?dense_35/MatMul/ReadVariableOp?
conv2d_11/Conv2D/ReadVariableOpReadVariableOp0conv2d_11_conv2d_readvariableop_conv2d_11_kernel*&
_output_shapes
:*
dtype02!
conv2d_11/Conv2D/ReadVariableOp?
conv2d_11/Conv2DConv2Dinputs'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_11/Conv2D?
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp/conv2d_11_biasadd_readvariableop_conv2d_11_bias*
_output_shapes
:*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp?
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_11/BiasAdd?
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_11/Relu?
max_pooling2d_10/MaxPoolMaxPoolconv2d_11/Relu:activations:0*/
_output_shapes
:?????????***
ksize
*
paddingVALID*
strides
2
max_pooling2d_10/MaxPool?
conv2d_12/Conv2D/ReadVariableOpReadVariableOp0conv2d_12_conv2d_readvariableop_conv2d_12_kernel*&
_output_shapes
: *
dtype02!
conv2d_12/Conv2D/ReadVariableOp?
conv2d_12/Conv2DConv2D!max_pooling2d_10/MaxPool:output:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(( *
paddingVALID*
strides
2
conv2d_12/Conv2D?
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp/conv2d_12_biasadd_readvariableop_conv2d_12_bias*
_output_shapes
: *
dtype02"
 conv2d_12/BiasAdd/ReadVariableOp?
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(( 2
conv2d_12/BiasAdd~
conv2d_12/ReluReluconv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(( 2
conv2d_12/Relu?
max_pooling2d_11/MaxPoolMaxPoolconv2d_12/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_11/MaxPool?
conv2d_13/Conv2D/ReadVariableOpReadVariableOp0conv2d_13_conv2d_readvariableop_conv2d_13_kernel*&
_output_shapes
: @*
dtype02!
conv2d_13/Conv2D/ReadVariableOp?
conv2d_13/Conv2DConv2D!max_pooling2d_11/MaxPool:output:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_13/Conv2D?
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp/conv2d_13_biasadd_readvariableop_conv2d_13_bias*
_output_shapes
:@*
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp?
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_13/BiasAdd~
conv2d_13/ReluReluconv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_13/Relu?
max_pooling2d_12/MaxPoolMaxPoolconv2d_13/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_12/MaxPools
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  2
flatten_3/Const?
flatten_3/ReshapeReshape!max_pooling2d_12/MaxPool:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_3/Reshape?
dense_27/MatMul/ReadVariableOpReadVariableOp.dense_27_matmul_readvariableop_dense_27_kernel* 
_output_shapes
:
??*
dtype02 
dense_27/MatMul/ReadVariableOp?
dense_27/MatMulMatMulflatten_3/Reshape:output:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_27/MatMul?
dense_27/BiasAdd/ReadVariableOpReadVariableOp-dense_27_biasadd_readvariableop_dense_27_bias*
_output_shapes	
:?*
dtype02!
dense_27/BiasAdd/ReadVariableOp?
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_27/BiasAddt
dense_27/ReluReludense_27/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_27/Relu?
dense_28/MatMul/ReadVariableOpReadVariableOp.dense_28_matmul_readvariableop_dense_28_kernel*
_output_shapes
:	?@*
dtype02 
dense_28/MatMul/ReadVariableOp?
dense_28/MatMulMatMuldense_27/Relu:activations:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_28/MatMul?
dense_28/BiasAdd/ReadVariableOpReadVariableOp-dense_28_biasadd_readvariableop_dense_28_bias*
_output_shapes
:@*
dtype02!
dense_28/BiasAdd/ReadVariableOp?
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_28/BiasAdds
dense_28/ReluReludense_28/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_28/Relu?
dense_29/MatMul/ReadVariableOpReadVariableOp.dense_29_matmul_readvariableop_dense_29_kernel*
_output_shapes

:@ *
dtype02 
dense_29/MatMul/ReadVariableOp?
dense_29/MatMulMatMuldense_28/Relu:activations:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_29/MatMul?
dense_29/BiasAdd/ReadVariableOpReadVariableOp-dense_29_biasadd_readvariableop_dense_29_bias*
_output_shapes
: *
dtype02!
dense_29/BiasAdd/ReadVariableOp?
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_29/BiasAdds
dense_29/ReluReludense_29/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_29/Relu?
dense_35/MatMul/ReadVariableOpReadVariableOp.dense_35_matmul_readvariableop_dense_35_kernel*
_output_shapes

: *
dtype02 
dense_35/MatMul/ReadVariableOp?
dense_35/MatMulMatMuldense_29/Relu:activations:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_35/MatMul?
dense_35/BiasAdd/ReadVariableOpReadVariableOp-dense_35_biasadd_readvariableop_dense_35_bias*
_output_shapes
:*
dtype02!
dense_35/BiasAdd/ReadVariableOp?
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_35/BiasAdd|
dense_35/SigmoidSigmoiddense_35/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_35/Sigmoid?
dense_34/MatMul/ReadVariableOpReadVariableOp.dense_34_matmul_readvariableop_dense_34_kernel*
_output_shapes

: *
dtype02 
dense_34/MatMul/ReadVariableOp?
dense_34/MatMulMatMuldense_29/Relu:activations:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_34/MatMul?
dense_34/BiasAdd/ReadVariableOpReadVariableOp-dense_34_biasadd_readvariableop_dense_34_bias*
_output_shapes
:*
dtype02!
dense_34/BiasAdd/ReadVariableOp?
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_34/BiasAdd|
dense_34/SigmoidSigmoiddense_34/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_34/Sigmoid?
dense_33/MatMul/ReadVariableOpReadVariableOp.dense_33_matmul_readvariableop_dense_33_kernel*
_output_shapes

: *
dtype02 
dense_33/MatMul/ReadVariableOp?
dense_33/MatMulMatMuldense_29/Relu:activations:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_33/MatMul?
dense_33/BiasAdd/ReadVariableOpReadVariableOp-dense_33_biasadd_readvariableop_dense_33_bias*
_output_shapes
:*
dtype02!
dense_33/BiasAdd/ReadVariableOp?
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_33/BiasAdd|
dense_33/SigmoidSigmoiddense_33/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_33/Sigmoid?
dense_32/MatMul/ReadVariableOpReadVariableOp.dense_32_matmul_readvariableop_dense_32_kernel*
_output_shapes

: *
dtype02 
dense_32/MatMul/ReadVariableOp?
dense_32/MatMulMatMuldense_29/Relu:activations:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_32/MatMul?
dense_32/BiasAdd/ReadVariableOpReadVariableOp-dense_32_biasadd_readvariableop_dense_32_bias*
_output_shapes
:*
dtype02!
dense_32/BiasAdd/ReadVariableOp?
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_32/BiasAdd|
dense_32/SigmoidSigmoiddense_32/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_32/Sigmoid?
dense_31/MatMul/ReadVariableOpReadVariableOp.dense_31_matmul_readvariableop_dense_31_kernel*
_output_shapes

: *
dtype02 
dense_31/MatMul/ReadVariableOp?
dense_31/MatMulMatMuldense_29/Relu:activations:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_31/MatMul?
dense_31/BiasAdd/ReadVariableOpReadVariableOp-dense_31_biasadd_readvariableop_dense_31_bias*
_output_shapes
:*
dtype02!
dense_31/BiasAdd/ReadVariableOp?
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_31/BiasAdd|
dense_31/SigmoidSigmoiddense_31/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_31/Sigmoid?
dense_30/MatMul/ReadVariableOpReadVariableOp.dense_30_matmul_readvariableop_dense_30_kernel*
_output_shapes

: *
dtype02 
dense_30/MatMul/ReadVariableOp?
dense_30/MatMulMatMuldense_29/Relu:activations:0&dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_30/MatMul?
dense_30/BiasAdd/ReadVariableOpReadVariableOp-dense_30_biasadd_readvariableop_dense_30_bias*
_output_shapes
:*
dtype02!
dense_30/BiasAdd/ReadVariableOp?
dense_30/BiasAddBiasAdddense_30/MatMul:product:0'dense_30/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_30/BiasAdd|
dense_30/SigmoidSigmoiddense_30/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_30/Sigmoid?
IdentityIdentitydense_30/Sigmoid:y:0!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitydense_31/Sigmoid:y:0!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identitydense_32/Sigmoid:y:0!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identitydense_33/Sigmoid:y:0!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identitydense_34/Sigmoid:y:0!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identitydense_35/Sigmoid:y:0!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*?
_input_shapes
}:???????????::::::::::::::::::::::::2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp2B
dense_30/BiasAdd/ReadVariableOpdense_30/BiasAdd/ReadVariableOp2@
dense_30/MatMul/ReadVariableOpdense_30/MatMul/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
C__inference_dense_32_layer_call_and_return_conditional_losses_11618

inputs)
%matmul_readvariableop_dense_32_kernel(
$biasadd_readvariableop_dense_32_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_32_kernel*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_32_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
D__inference_conv2d_12_layer_call_and_return_conditional_losses_10637

inputs*
&conv2d_readvariableop_conv2d_12_kernel)
%biasadd_readvariableop_conv2d_12_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp&conv2d_readvariableop_conv2d_12_kernel*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(( *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_conv2d_12_bias*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(( 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????(( 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????(( 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????**::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????**
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_11_layer_call_fn_10576

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_105732
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
C__inference_dense_27_layer_call_and_return_conditional_losses_10709

inputs)
%matmul_readvariableop_dense_27_kernel(
$biasadd_readvariableop_dense_27_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_27_kernel* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_27_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?2
!__inference__traced_restore_12251
file_prefix%
!assignvariableop_conv2d_11_kernel%
!assignvariableop_1_conv2d_11_bias'
#assignvariableop_2_conv2d_12_kernel%
!assignvariableop_3_conv2d_12_bias'
#assignvariableop_4_conv2d_13_kernel%
!assignvariableop_5_conv2d_13_bias&
"assignvariableop_6_dense_27_kernel$
 assignvariableop_7_dense_27_bias&
"assignvariableop_8_dense_28_kernel$
 assignvariableop_9_dense_28_bias'
#assignvariableop_10_dense_29_kernel%
!assignvariableop_11_dense_29_bias'
#assignvariableop_12_dense_30_kernel%
!assignvariableop_13_dense_30_bias'
#assignvariableop_14_dense_31_kernel%
!assignvariableop_15_dense_31_bias'
#assignvariableop_16_dense_32_kernel%
!assignvariableop_17_dense_32_bias'
#assignvariableop_18_dense_33_kernel%
!assignvariableop_19_dense_33_bias'
#assignvariableop_20_dense_34_kernel%
!assignvariableop_21_dense_34_bias'
#assignvariableop_22_dense_35_kernel%
!assignvariableop_23_dense_35_bias,
(assignvariableop_24_training_8_adam_iter.
*assignvariableop_25_training_8_adam_beta_1.
*assignvariableop_26_training_8_adam_beta_2-
)assignvariableop_27_training_8_adam_decay5
1assignvariableop_28_training_8_adam_learning_rate 
assignvariableop_29_total_48 
assignvariableop_30_count_48 
assignvariableop_31_total_49 
assignvariableop_32_count_49 
assignvariableop_33_total_50 
assignvariableop_34_count_50 
assignvariableop_35_total_51 
assignvariableop_36_count_51 
assignvariableop_37_total_52 
assignvariableop_38_count_52 
assignvariableop_39_total_53 
assignvariableop_40_count_53:
6assignvariableop_41_training_8_adam_conv2d_11_kernel_m8
4assignvariableop_42_training_8_adam_conv2d_11_bias_m:
6assignvariableop_43_training_8_adam_conv2d_12_kernel_m8
4assignvariableop_44_training_8_adam_conv2d_12_bias_m:
6assignvariableop_45_training_8_adam_conv2d_13_kernel_m8
4assignvariableop_46_training_8_adam_conv2d_13_bias_m9
5assignvariableop_47_training_8_adam_dense_27_kernel_m7
3assignvariableop_48_training_8_adam_dense_27_bias_m9
5assignvariableop_49_training_8_adam_dense_28_kernel_m7
3assignvariableop_50_training_8_adam_dense_28_bias_m9
5assignvariableop_51_training_8_adam_dense_29_kernel_m7
3assignvariableop_52_training_8_adam_dense_29_bias_m9
5assignvariableop_53_training_8_adam_dense_30_kernel_m7
3assignvariableop_54_training_8_adam_dense_30_bias_m9
5assignvariableop_55_training_8_adam_dense_31_kernel_m7
3assignvariableop_56_training_8_adam_dense_31_bias_m9
5assignvariableop_57_training_8_adam_dense_32_kernel_m7
3assignvariableop_58_training_8_adam_dense_32_bias_m9
5assignvariableop_59_training_8_adam_dense_33_kernel_m7
3assignvariableop_60_training_8_adam_dense_33_bias_m9
5assignvariableop_61_training_8_adam_dense_34_kernel_m7
3assignvariableop_62_training_8_adam_dense_34_bias_m9
5assignvariableop_63_training_8_adam_dense_35_kernel_m7
3assignvariableop_64_training_8_adam_dense_35_bias_m:
6assignvariableop_65_training_8_adam_conv2d_11_kernel_v8
4assignvariableop_66_training_8_adam_conv2d_11_bias_v:
6assignvariableop_67_training_8_adam_conv2d_12_kernel_v8
4assignvariableop_68_training_8_adam_conv2d_12_bias_v:
6assignvariableop_69_training_8_adam_conv2d_13_kernel_v8
4assignvariableop_70_training_8_adam_conv2d_13_bias_v9
5assignvariableop_71_training_8_adam_dense_27_kernel_v7
3assignvariableop_72_training_8_adam_dense_27_bias_v9
5assignvariableop_73_training_8_adam_dense_28_kernel_v7
3assignvariableop_74_training_8_adam_dense_28_bias_v9
5assignvariableop_75_training_8_adam_dense_29_kernel_v7
3assignvariableop_76_training_8_adam_dense_29_bias_v9
5assignvariableop_77_training_8_adam_dense_30_kernel_v7
3assignvariableop_78_training_8_adam_dense_30_bias_v9
5assignvariableop_79_training_8_adam_dense_31_kernel_v7
3assignvariableop_80_training_8_adam_dense_31_bias_v9
5assignvariableop_81_training_8_adam_dense_32_kernel_v7
3assignvariableop_82_training_8_adam_dense_32_bias_v9
5assignvariableop_83_training_8_adam_dense_33_kernel_v7
3assignvariableop_84_training_8_adam_dense_33_bias_v9
5assignvariableop_85_training_8_adam_dense_34_kernel_v7
3assignvariableop_86_training_8_adam_dense_34_bias_v9
5assignvariableop_87_training_8_adam_dense_35_kernel_v7
3assignvariableop_88_training_8_adam_dense_35_bias_v
identity_90??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_9?1
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Z*
dtype0*?0
value?0B?0ZB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Z*
dtype0*?
value?B?ZB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*h
dtypes^
\2Z	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_11_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_11_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_12_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_12_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_13_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_13_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_27_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_27_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_28_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_28_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_29_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_29_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_30_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_30_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_31_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_31_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_32_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_32_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp#assignvariableop_18_dense_33_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp!assignvariableop_19_dense_33_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp#assignvariableop_20_dense_34_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp!assignvariableop_21_dense_34_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp#assignvariableop_22_dense_35_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp!assignvariableop_23_dense_35_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp(assignvariableop_24_training_8_adam_iterIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_training_8_adam_beta_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp*assignvariableop_26_training_8_adam_beta_2Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp)assignvariableop_27_training_8_adam_decayIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp1assignvariableop_28_training_8_adam_learning_rateIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpassignvariableop_29_total_48Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpassignvariableop_30_count_48Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpassignvariableop_31_total_49Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOpassignvariableop_32_count_49Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOpassignvariableop_33_total_50Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOpassignvariableop_34_count_50Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOpassignvariableop_35_total_51Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOpassignvariableop_36_count_51Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOpassignvariableop_37_total_52Identity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOpassignvariableop_38_count_52Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOpassignvariableop_39_total_53Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOpassignvariableop_40_count_53Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp6assignvariableop_41_training_8_adam_conv2d_11_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp4assignvariableop_42_training_8_adam_conv2d_11_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp6assignvariableop_43_training_8_adam_conv2d_12_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp4assignvariableop_44_training_8_adam_conv2d_12_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp6assignvariableop_45_training_8_adam_conv2d_13_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp4assignvariableop_46_training_8_adam_conv2d_13_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp5assignvariableop_47_training_8_adam_dense_27_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp3assignvariableop_48_training_8_adam_dense_27_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp5assignvariableop_49_training_8_adam_dense_28_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp3assignvariableop_50_training_8_adam_dense_28_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp5assignvariableop_51_training_8_adam_dense_29_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp3assignvariableop_52_training_8_adam_dense_29_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp5assignvariableop_53_training_8_adam_dense_30_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp3assignvariableop_54_training_8_adam_dense_30_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp5assignvariableop_55_training_8_adam_dense_31_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp3assignvariableop_56_training_8_adam_dense_31_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp5assignvariableop_57_training_8_adam_dense_32_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp3assignvariableop_58_training_8_adam_dense_32_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp5assignvariableop_59_training_8_adam_dense_33_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp3assignvariableop_60_training_8_adam_dense_33_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp5assignvariableop_61_training_8_adam_dense_34_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp3assignvariableop_62_training_8_adam_dense_34_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp5assignvariableop_63_training_8_adam_dense_35_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp3assignvariableop_64_training_8_adam_dense_35_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp6assignvariableop_65_training_8_adam_conv2d_11_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp4assignvariableop_66_training_8_adam_conv2d_11_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp6assignvariableop_67_training_8_adam_conv2d_12_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp4assignvariableop_68_training_8_adam_conv2d_12_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp6assignvariableop_69_training_8_adam_conv2d_13_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp4assignvariableop_70_training_8_adam_conv2d_13_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp5assignvariableop_71_training_8_adam_dense_27_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp3assignvariableop_72_training_8_adam_dense_27_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp5assignvariableop_73_training_8_adam_dense_28_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp3assignvariableop_74_training_8_adam_dense_28_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp5assignvariableop_75_training_8_adam_dense_29_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp3assignvariableop_76_training_8_adam_dense_29_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp5assignvariableop_77_training_8_adam_dense_30_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp3assignvariableop_78_training_8_adam_dense_30_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp5assignvariableop_79_training_8_adam_dense_31_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp3assignvariableop_80_training_8_adam_dense_31_bias_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp5assignvariableop_81_training_8_adam_dense_32_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp3assignvariableop_82_training_8_adam_dense_32_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOp5assignvariableop_83_training_8_adam_dense_33_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp3assignvariableop_84_training_8_adam_dense_33_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOp5assignvariableop_85_training_8_adam_dense_34_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOp3assignvariableop_86_training_8_adam_dense_34_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOp5assignvariableop_87_training_8_adam_dense_35_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOp3assignvariableop_88_training_8_adam_dense_35_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_889
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_89Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_89?
Identity_90IdentityIdentity_89:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_90"#
identity_90Identity_90:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ס
?
B__inference_model_3_layer_call_and_return_conditional_losses_11374

inputs4
0conv2d_11_conv2d_readvariableop_conv2d_11_kernel3
/conv2d_11_biasadd_readvariableop_conv2d_11_bias4
0conv2d_12_conv2d_readvariableop_conv2d_12_kernel3
/conv2d_12_biasadd_readvariableop_conv2d_12_bias4
0conv2d_13_conv2d_readvariableop_conv2d_13_kernel3
/conv2d_13_biasadd_readvariableop_conv2d_13_bias2
.dense_27_matmul_readvariableop_dense_27_kernel1
-dense_27_biasadd_readvariableop_dense_27_bias2
.dense_28_matmul_readvariableop_dense_28_kernel1
-dense_28_biasadd_readvariableop_dense_28_bias2
.dense_29_matmul_readvariableop_dense_29_kernel1
-dense_29_biasadd_readvariableop_dense_29_bias2
.dense_35_matmul_readvariableop_dense_35_kernel1
-dense_35_biasadd_readvariableop_dense_35_bias2
.dense_34_matmul_readvariableop_dense_34_kernel1
-dense_34_biasadd_readvariableop_dense_34_bias2
.dense_33_matmul_readvariableop_dense_33_kernel1
-dense_33_biasadd_readvariableop_dense_33_bias2
.dense_32_matmul_readvariableop_dense_32_kernel1
-dense_32_biasadd_readvariableop_dense_32_bias2
.dense_31_matmul_readvariableop_dense_31_kernel1
-dense_31_biasadd_readvariableop_dense_31_bias2
.dense_30_matmul_readvariableop_dense_30_kernel1
-dense_30_biasadd_readvariableop_dense_30_bias
identity

identity_1

identity_2

identity_3

identity_4

identity_5?? conv2d_11/BiasAdd/ReadVariableOp?conv2d_11/Conv2D/ReadVariableOp? conv2d_12/BiasAdd/ReadVariableOp?conv2d_12/Conv2D/ReadVariableOp? conv2d_13/BiasAdd/ReadVariableOp?conv2d_13/Conv2D/ReadVariableOp?dense_27/BiasAdd/ReadVariableOp?dense_27/MatMul/ReadVariableOp?dense_28/BiasAdd/ReadVariableOp?dense_28/MatMul/ReadVariableOp?dense_29/BiasAdd/ReadVariableOp?dense_29/MatMul/ReadVariableOp?dense_30/BiasAdd/ReadVariableOp?dense_30/MatMul/ReadVariableOp?dense_31/BiasAdd/ReadVariableOp?dense_31/MatMul/ReadVariableOp?dense_32/BiasAdd/ReadVariableOp?dense_32/MatMul/ReadVariableOp?dense_33/BiasAdd/ReadVariableOp?dense_33/MatMul/ReadVariableOp?dense_34/BiasAdd/ReadVariableOp?dense_34/MatMul/ReadVariableOp?dense_35/BiasAdd/ReadVariableOp?dense_35/MatMul/ReadVariableOp?
conv2d_11/Conv2D/ReadVariableOpReadVariableOp0conv2d_11_conv2d_readvariableop_conv2d_11_kernel*&
_output_shapes
:*
dtype02!
conv2d_11/Conv2D/ReadVariableOp?
conv2d_11/Conv2DConv2Dinputs'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_11/Conv2D?
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp/conv2d_11_biasadd_readvariableop_conv2d_11_bias*
_output_shapes
:*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp?
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_11/BiasAdd?
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_11/Relu?
max_pooling2d_10/MaxPoolMaxPoolconv2d_11/Relu:activations:0*/
_output_shapes
:?????????***
ksize
*
paddingVALID*
strides
2
max_pooling2d_10/MaxPool?
conv2d_12/Conv2D/ReadVariableOpReadVariableOp0conv2d_12_conv2d_readvariableop_conv2d_12_kernel*&
_output_shapes
: *
dtype02!
conv2d_12/Conv2D/ReadVariableOp?
conv2d_12/Conv2DConv2D!max_pooling2d_10/MaxPool:output:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(( *
paddingVALID*
strides
2
conv2d_12/Conv2D?
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp/conv2d_12_biasadd_readvariableop_conv2d_12_bias*
_output_shapes
: *
dtype02"
 conv2d_12/BiasAdd/ReadVariableOp?
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(( 2
conv2d_12/BiasAdd~
conv2d_12/ReluReluconv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(( 2
conv2d_12/Relu?
max_pooling2d_11/MaxPoolMaxPoolconv2d_12/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_11/MaxPool?
conv2d_13/Conv2D/ReadVariableOpReadVariableOp0conv2d_13_conv2d_readvariableop_conv2d_13_kernel*&
_output_shapes
: @*
dtype02!
conv2d_13/Conv2D/ReadVariableOp?
conv2d_13/Conv2DConv2D!max_pooling2d_11/MaxPool:output:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_13/Conv2D?
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp/conv2d_13_biasadd_readvariableop_conv2d_13_bias*
_output_shapes
:@*
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp?
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_13/BiasAdd~
conv2d_13/ReluReluconv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_13/Relu?
max_pooling2d_12/MaxPoolMaxPoolconv2d_13/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_12/MaxPools
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  2
flatten_3/Const?
flatten_3/ReshapeReshape!max_pooling2d_12/MaxPool:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_3/Reshape?
dense_27/MatMul/ReadVariableOpReadVariableOp.dense_27_matmul_readvariableop_dense_27_kernel* 
_output_shapes
:
??*
dtype02 
dense_27/MatMul/ReadVariableOp?
dense_27/MatMulMatMulflatten_3/Reshape:output:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_27/MatMul?
dense_27/BiasAdd/ReadVariableOpReadVariableOp-dense_27_biasadd_readvariableop_dense_27_bias*
_output_shapes	
:?*
dtype02!
dense_27/BiasAdd/ReadVariableOp?
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_27/BiasAddt
dense_27/ReluReludense_27/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_27/Relu?
dense_28/MatMul/ReadVariableOpReadVariableOp.dense_28_matmul_readvariableop_dense_28_kernel*
_output_shapes
:	?@*
dtype02 
dense_28/MatMul/ReadVariableOp?
dense_28/MatMulMatMuldense_27/Relu:activations:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_28/MatMul?
dense_28/BiasAdd/ReadVariableOpReadVariableOp-dense_28_biasadd_readvariableop_dense_28_bias*
_output_shapes
:@*
dtype02!
dense_28/BiasAdd/ReadVariableOp?
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_28/BiasAdds
dense_28/ReluReludense_28/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_28/Relu?
dense_29/MatMul/ReadVariableOpReadVariableOp.dense_29_matmul_readvariableop_dense_29_kernel*
_output_shapes

:@ *
dtype02 
dense_29/MatMul/ReadVariableOp?
dense_29/MatMulMatMuldense_28/Relu:activations:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_29/MatMul?
dense_29/BiasAdd/ReadVariableOpReadVariableOp-dense_29_biasadd_readvariableop_dense_29_bias*
_output_shapes
: *
dtype02!
dense_29/BiasAdd/ReadVariableOp?
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_29/BiasAdds
dense_29/ReluReludense_29/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_29/Relu?
dense_35/MatMul/ReadVariableOpReadVariableOp.dense_35_matmul_readvariableop_dense_35_kernel*
_output_shapes

: *
dtype02 
dense_35/MatMul/ReadVariableOp?
dense_35/MatMulMatMuldense_29/Relu:activations:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_35/MatMul?
dense_35/BiasAdd/ReadVariableOpReadVariableOp-dense_35_biasadd_readvariableop_dense_35_bias*
_output_shapes
:*
dtype02!
dense_35/BiasAdd/ReadVariableOp?
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_35/BiasAdd|
dense_35/SigmoidSigmoiddense_35/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_35/Sigmoid?
dense_34/MatMul/ReadVariableOpReadVariableOp.dense_34_matmul_readvariableop_dense_34_kernel*
_output_shapes

: *
dtype02 
dense_34/MatMul/ReadVariableOp?
dense_34/MatMulMatMuldense_29/Relu:activations:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_34/MatMul?
dense_34/BiasAdd/ReadVariableOpReadVariableOp-dense_34_biasadd_readvariableop_dense_34_bias*
_output_shapes
:*
dtype02!
dense_34/BiasAdd/ReadVariableOp?
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_34/BiasAdd|
dense_34/SigmoidSigmoiddense_34/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_34/Sigmoid?
dense_33/MatMul/ReadVariableOpReadVariableOp.dense_33_matmul_readvariableop_dense_33_kernel*
_output_shapes

: *
dtype02 
dense_33/MatMul/ReadVariableOp?
dense_33/MatMulMatMuldense_29/Relu:activations:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_33/MatMul?
dense_33/BiasAdd/ReadVariableOpReadVariableOp-dense_33_biasadd_readvariableop_dense_33_bias*
_output_shapes
:*
dtype02!
dense_33/BiasAdd/ReadVariableOp?
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_33/BiasAdd|
dense_33/SigmoidSigmoiddense_33/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_33/Sigmoid?
dense_32/MatMul/ReadVariableOpReadVariableOp.dense_32_matmul_readvariableop_dense_32_kernel*
_output_shapes

: *
dtype02 
dense_32/MatMul/ReadVariableOp?
dense_32/MatMulMatMuldense_29/Relu:activations:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_32/MatMul?
dense_32/BiasAdd/ReadVariableOpReadVariableOp-dense_32_biasadd_readvariableop_dense_32_bias*
_output_shapes
:*
dtype02!
dense_32/BiasAdd/ReadVariableOp?
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_32/BiasAdd|
dense_32/SigmoidSigmoiddense_32/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_32/Sigmoid?
dense_31/MatMul/ReadVariableOpReadVariableOp.dense_31_matmul_readvariableop_dense_31_kernel*
_output_shapes

: *
dtype02 
dense_31/MatMul/ReadVariableOp?
dense_31/MatMulMatMuldense_29/Relu:activations:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_31/MatMul?
dense_31/BiasAdd/ReadVariableOpReadVariableOp-dense_31_biasadd_readvariableop_dense_31_bias*
_output_shapes
:*
dtype02!
dense_31/BiasAdd/ReadVariableOp?
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_31/BiasAdd|
dense_31/SigmoidSigmoiddense_31/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_31/Sigmoid?
dense_30/MatMul/ReadVariableOpReadVariableOp.dense_30_matmul_readvariableop_dense_30_kernel*
_output_shapes

: *
dtype02 
dense_30/MatMul/ReadVariableOp?
dense_30/MatMulMatMuldense_29/Relu:activations:0&dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_30/MatMul?
dense_30/BiasAdd/ReadVariableOpReadVariableOp-dense_30_biasadd_readvariableop_dense_30_bias*
_output_shapes
:*
dtype02!
dense_30/BiasAdd/ReadVariableOp?
dense_30/BiasAddBiasAdddense_30/MatMul:product:0'dense_30/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_30/BiasAdd|
dense_30/SigmoidSigmoiddense_30/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_30/Sigmoid?
IdentityIdentitydense_30/Sigmoid:y:0!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitydense_31/Sigmoid:y:0!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identitydense_32/Sigmoid:y:0!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identitydense_33/Sigmoid:y:0!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identitydense_34/Sigmoid:y:0!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identitydense_35/Sigmoid:y:0!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*?
_input_shapes
}:???????????::::::::::::::::::::::::2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp2B
dense_30/BiasAdd/ReadVariableOpdense_30/BiasAdd/ReadVariableOp2@
dense_30/MatMul/ReadVariableOpdense_30/MatMul/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
C__inference_dense_31_layer_call_and_return_conditional_losses_10870

inputs)
%matmul_readvariableop_dense_31_kernel(
$biasadd_readvariableop_dense_31_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_31_kernel*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_31_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
C__inference_dense_29_layer_call_and_return_conditional_losses_11564

inputs)
%matmul_readvariableop_dense_29_kernel(
$biasadd_readvariableop_dense_29_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_29_kernel*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_29_bias*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
)__inference_conv2d_11_layer_call_fn_11470

inputs
conv2d_11_kernel
conv2d_11_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_11_kernelconv2d_11_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_106082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_11_layer_call_and_return_conditional_losses_10608

inputs*
&conv2d_readvariableop_conv2d_11_kernel)
%biasadd_readvariableop_conv2d_11_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp&conv2d_readvariableop_conv2d_11_kernel*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_conv2d_11_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_10590

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?e
?

B__inference_model_3_layer_call_and_return_conditional_losses_11012

inputs
conv2d_11_conv2d_11_kernel
conv2d_11_conv2d_11_bias
conv2d_12_conv2d_12_kernel
conv2d_12_conv2d_12_bias
conv2d_13_conv2d_13_kernel
conv2d_13_conv2d_13_bias
dense_27_dense_27_kernel
dense_27_dense_27_bias
dense_28_dense_28_kernel
dense_28_dense_28_bias
dense_29_dense_29_kernel
dense_29_dense_29_bias
dense_35_dense_35_kernel
dense_35_dense_35_bias
dense_34_dense_34_kernel
dense_34_dense_34_bias
dense_33_dense_33_kernel
dense_33_dense_33_bias
dense_32_dense_32_kernel
dense_32_dense_32_bias
dense_31_dense_31_kernel
dense_31_dense_31_bias
dense_30_dense_30_kernel
dense_30_dense_30_bias
identity

identity_1

identity_2

identity_3

identity_4

identity_5??!conv2d_11/StatefulPartitionedCall?!conv2d_12/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall? dense_27/StatefulPartitionedCall? dense_28/StatefulPartitionedCall? dense_29/StatefulPartitionedCall? dense_30/StatefulPartitionedCall? dense_31/StatefulPartitionedCall? dense_32/StatefulPartitionedCall? dense_33/StatefulPartitionedCall? dense_34/StatefulPartitionedCall? dense_35/StatefulPartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_11_conv2d_11_kernelconv2d_11_conv2d_11_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_106082#
!conv2d_11/StatefulPartitionedCall?
 max_pooling2d_10/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_105562"
 max_pooling2d_10/PartitionedCall?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_10/PartitionedCall:output:0conv2d_12_conv2d_12_kernelconv2d_12_conv2d_12_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(( *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_106372#
!conv2d_12/StatefulPartitionedCall?
 max_pooling2d_11/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_105732"
 max_pooling2d_11/PartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_11/PartitionedCall:output:0conv2d_13_conv2d_13_kernelconv2d_13_conv2d_13_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_106662#
!conv2d_13/StatefulPartitionedCall?
 max_pooling2d_12/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_105902"
 max_pooling2d_12/PartitionedCall?
flatten_3/PartitionedCallPartitionedCall)max_pooling2d_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_106902
flatten_3/PartitionedCall?
 dense_27/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_27_dense_27_kerneldense_27_dense_27_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_27_layer_call_and_return_conditional_losses_107092"
 dense_27/StatefulPartitionedCall?
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0dense_28_dense_28_kerneldense_28_dense_28_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_28_layer_call_and_return_conditional_losses_107322"
 dense_28/StatefulPartitionedCall?
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_dense_29_kerneldense_29_dense_29_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_29_layer_call_and_return_conditional_losses_107552"
 dense_29/StatefulPartitionedCall?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_29/StatefulPartitionedCall:output:0dense_35_dense_35_kerneldense_35_dense_35_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_35_layer_call_and_return_conditional_losses_107782"
 dense_35/StatefulPartitionedCall?
 dense_34/StatefulPartitionedCallStatefulPartitionedCall)dense_29/StatefulPartitionedCall:output:0dense_34_dense_34_kerneldense_34_dense_34_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_34_layer_call_and_return_conditional_losses_108012"
 dense_34/StatefulPartitionedCall?
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_29/StatefulPartitionedCall:output:0dense_33_dense_33_kerneldense_33_dense_33_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_33_layer_call_and_return_conditional_losses_108242"
 dense_33/StatefulPartitionedCall?
 dense_32/StatefulPartitionedCallStatefulPartitionedCall)dense_29/StatefulPartitionedCall:output:0dense_32_dense_32_kerneldense_32_dense_32_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_32_layer_call_and_return_conditional_losses_108472"
 dense_32/StatefulPartitionedCall?
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_29/StatefulPartitionedCall:output:0dense_31_dense_31_kerneldense_31_dense_31_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_31_layer_call_and_return_conditional_losses_108702"
 dense_31/StatefulPartitionedCall?
 dense_30/StatefulPartitionedCallStatefulPartitionedCall)dense_29/StatefulPartitionedCall:output:0dense_30_dense_30_kerneldense_30_dense_30_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_30_layer_call_and_return_conditional_losses_108932"
 dense_30/StatefulPartitionedCall?
IdentityIdentity)dense_30/StatefulPartitionedCall:output:0"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity)dense_31/StatefulPartitionedCall:output:0"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity)dense_32/StatefulPartitionedCall:output:0"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity)dense_33/StatefulPartitionedCall:output:0"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity)dense_34/StatefulPartitionedCall:output:0"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity)dense_35/StatefulPartitionedCall:output:0"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*?
_input_shapes
}:???????????::::::::::::::::::::::::2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
(__inference_dense_30_layer_call_fn_11589

inputs
dense_30_kernel
dense_30_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_30_kerneldense_30_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_30_layer_call_and_return_conditional_losses_108932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
C__inference_dense_30_layer_call_and_return_conditional_losses_11582

inputs)
%matmul_readvariableop_dense_30_kernel(
$biasadd_readvariableop_dense_30_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_30_kernel*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_30_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
'__inference_model_3_layer_call_fn_11137
input_4
conv2d_11_kernel
conv2d_11_bias
conv2d_12_kernel
conv2d_12_bias
conv2d_13_kernel
conv2d_13_bias
dense_27_kernel
dense_27_bias
dense_28_kernel
dense_28_bias
dense_29_kernel
dense_29_bias
dense_35_kernel
dense_35_bias
dense_34_kernel
dense_34_bias
dense_33_kernel
dense_33_bias
dense_32_kernel
dense_32_bias
dense_31_kernel
dense_31_bias
dense_30_kernel
dense_30_bias
identity

identity_1

identity_2

identity_3

identity_4

identity_5??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_4conv2d_11_kernelconv2d_11_biasconv2d_12_kernelconv2d_12_biasconv2d_13_kernelconv2d_13_biasdense_27_kerneldense_27_biasdense_28_kerneldense_28_biasdense_29_kerneldense_29_biasdense_35_kerneldense_35_biasdense_34_kerneldense_34_biasdense_33_kerneldense_33_biasdense_32_kerneldense_32_biasdense_31_kerneldense_31_biasdense_30_kerneldense_30_bias*$
Tin
2*
Tout

2*
_collective_manager_ids
 *?
_output_shapest
r:?????????:?????????:?????????:?????????:?????????:?????????*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_111002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*?
_input_shapes
}:???????????::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_4
?
?
(__inference_dense_29_layer_call_fn_11571

inputs
dense_29_kernel
dense_29_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_29_kerneldense_29_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_29_layer_call_and_return_conditional_losses_107552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
C__inference_dense_33_layer_call_and_return_conditional_losses_11636

inputs)
%matmul_readvariableop_dense_33_kernel(
$biasadd_readvariableop_dense_33_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_33_kernel*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_33_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_10573

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
C__inference_dense_28_layer_call_and_return_conditional_losses_11546

inputs)
%matmul_readvariableop_dense_28_kernel(
$biasadd_readvariableop_dense_28_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_28_kernel*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_28_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_13_layer_call_and_return_conditional_losses_11499

inputs*
&conv2d_readvariableop_conv2d_13_kernel)
%biasadd_readvariableop_conv2d_13_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp&conv2d_readvariableop_conv2d_13_kernel*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_conv2d_13_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
E
)__inference_flatten_3_layer_call_fn_11517

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_106902
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
(__inference_dense_35_layer_call_fn_11679

inputs
dense_35_kernel
dense_35_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_35_kerneldense_35_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_35_layer_call_and_return_conditional_losses_107782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
'__inference_model_3_layer_call_fn_11049
input_4
conv2d_11_kernel
conv2d_11_bias
conv2d_12_kernel
conv2d_12_bias
conv2d_13_kernel
conv2d_13_bias
dense_27_kernel
dense_27_bias
dense_28_kernel
dense_28_bias
dense_29_kernel
dense_29_bias
dense_35_kernel
dense_35_bias
dense_34_kernel
dense_34_bias
dense_33_kernel
dense_33_bias
dense_32_kernel
dense_32_bias
dense_31_kernel
dense_31_bias
dense_30_kernel
dense_30_bias
identity

identity_1

identity_2

identity_3

identity_4

identity_5??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_4conv2d_11_kernelconv2d_11_biasconv2d_12_kernelconv2d_12_biasconv2d_13_kernelconv2d_13_biasdense_27_kerneldense_27_biasdense_28_kerneldense_28_biasdense_29_kerneldense_29_biasdense_35_kerneldense_35_biasdense_34_kerneldense_34_biasdense_33_kerneldense_33_biasdense_32_kerneldense_32_biasdense_31_kerneldense_31_biasdense_30_kerneldense_30_bias*$
Tin
2*
Tout

2*
_collective_manager_ids
 *?
_output_shapest
r:?????????:?????????:?????????:?????????:?????????:?????????*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_110122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*?
_input_shapes
}:???????????::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_4
?e
?

B__inference_model_3_layer_call_and_return_conditional_losses_11100

inputs
conv2d_11_conv2d_11_kernel
conv2d_11_conv2d_11_bias
conv2d_12_conv2d_12_kernel
conv2d_12_conv2d_12_bias
conv2d_13_conv2d_13_kernel
conv2d_13_conv2d_13_bias
dense_27_dense_27_kernel
dense_27_dense_27_bias
dense_28_dense_28_kernel
dense_28_dense_28_bias
dense_29_dense_29_kernel
dense_29_dense_29_bias
dense_35_dense_35_kernel
dense_35_dense_35_bias
dense_34_dense_34_kernel
dense_34_dense_34_bias
dense_33_dense_33_kernel
dense_33_dense_33_bias
dense_32_dense_32_kernel
dense_32_dense_32_bias
dense_31_dense_31_kernel
dense_31_dense_31_bias
dense_30_dense_30_kernel
dense_30_dense_30_bias
identity

identity_1

identity_2

identity_3

identity_4

identity_5??!conv2d_11/StatefulPartitionedCall?!conv2d_12/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall? dense_27/StatefulPartitionedCall? dense_28/StatefulPartitionedCall? dense_29/StatefulPartitionedCall? dense_30/StatefulPartitionedCall? dense_31/StatefulPartitionedCall? dense_32/StatefulPartitionedCall? dense_33/StatefulPartitionedCall? dense_34/StatefulPartitionedCall? dense_35/StatefulPartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_11_conv2d_11_kernelconv2d_11_conv2d_11_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_106082#
!conv2d_11/StatefulPartitionedCall?
 max_pooling2d_10/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_105562"
 max_pooling2d_10/PartitionedCall?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_10/PartitionedCall:output:0conv2d_12_conv2d_12_kernelconv2d_12_conv2d_12_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(( *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_106372#
!conv2d_12/StatefulPartitionedCall?
 max_pooling2d_11/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_105732"
 max_pooling2d_11/PartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_11/PartitionedCall:output:0conv2d_13_conv2d_13_kernelconv2d_13_conv2d_13_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_106662#
!conv2d_13/StatefulPartitionedCall?
 max_pooling2d_12/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_105902"
 max_pooling2d_12/PartitionedCall?
flatten_3/PartitionedCallPartitionedCall)max_pooling2d_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_106902
flatten_3/PartitionedCall?
 dense_27/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_27_dense_27_kerneldense_27_dense_27_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_27_layer_call_and_return_conditional_losses_107092"
 dense_27/StatefulPartitionedCall?
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0dense_28_dense_28_kerneldense_28_dense_28_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_28_layer_call_and_return_conditional_losses_107322"
 dense_28/StatefulPartitionedCall?
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_dense_29_kerneldense_29_dense_29_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_29_layer_call_and_return_conditional_losses_107552"
 dense_29/StatefulPartitionedCall?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_29/StatefulPartitionedCall:output:0dense_35_dense_35_kerneldense_35_dense_35_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_35_layer_call_and_return_conditional_losses_107782"
 dense_35/StatefulPartitionedCall?
 dense_34/StatefulPartitionedCallStatefulPartitionedCall)dense_29/StatefulPartitionedCall:output:0dense_34_dense_34_kerneldense_34_dense_34_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_34_layer_call_and_return_conditional_losses_108012"
 dense_34/StatefulPartitionedCall?
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_29/StatefulPartitionedCall:output:0dense_33_dense_33_kerneldense_33_dense_33_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_33_layer_call_and_return_conditional_losses_108242"
 dense_33/StatefulPartitionedCall?
 dense_32/StatefulPartitionedCallStatefulPartitionedCall)dense_29/StatefulPartitionedCall:output:0dense_32_dense_32_kerneldense_32_dense_32_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_32_layer_call_and_return_conditional_losses_108472"
 dense_32/StatefulPartitionedCall?
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_29/StatefulPartitionedCall:output:0dense_31_dense_31_kerneldense_31_dense_31_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_31_layer_call_and_return_conditional_losses_108702"
 dense_31/StatefulPartitionedCall?
 dense_30/StatefulPartitionedCallStatefulPartitionedCall)dense_29/StatefulPartitionedCall:output:0dense_30_dense_30_kerneldense_30_dense_30_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_30_layer_call_and_return_conditional_losses_108932"
 dense_30/StatefulPartitionedCall?
IdentityIdentity)dense_30/StatefulPartitionedCall:output:0"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity)dense_31/StatefulPartitionedCall:output:0"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity)dense_32/StatefulPartitionedCall:output:0"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity)dense_33/StatefulPartitionedCall:output:0"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity)dense_34/StatefulPartitionedCall:output:0"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity)dense_35/StatefulPartitionedCall:output:0"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*?
_input_shapes
}:???????????::::::::::::::::::::::::2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_10556

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
)__inference_conv2d_12_layer_call_fn_11488

inputs
conv2d_12_kernel
conv2d_12_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_12_kernelconv2d_12_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(( *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_106372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????(( 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????**::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????**
 
_user_specified_nameinputs
?
?
(__inference_dense_28_layer_call_fn_11553

inputs
dense_28_kernel
dense_28_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_28_kerneldense_28_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_28_layer_call_and_return_conditional_losses_107322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_11178
input_4
conv2d_11_kernel
conv2d_11_bias
conv2d_12_kernel
conv2d_12_bias
conv2d_13_kernel
conv2d_13_bias
dense_27_kernel
dense_27_bias
dense_28_kernel
dense_28_bias
dense_29_kernel
dense_29_bias
dense_35_kernel
dense_35_bias
dense_34_kernel
dense_34_bias
dense_33_kernel
dense_33_bias
dense_32_kernel
dense_32_bias
dense_31_kernel
dense_31_bias
dense_30_kernel
dense_30_bias
identity

identity_1

identity_2

identity_3

identity_4

identity_5??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_4conv2d_11_kernelconv2d_11_biasconv2d_12_kernelconv2d_12_biasconv2d_13_kernelconv2d_13_biasdense_27_kerneldense_27_biasdense_28_kerneldense_28_biasdense_29_kerneldense_29_biasdense_35_kerneldense_35_biasdense_34_kerneldense_34_biasdense_33_kerneldense_33_biasdense_32_kerneldense_32_biasdense_31_kerneldense_31_biasdense_30_kerneldense_30_bias*$
Tin
2*
Tout

2*
_collective_manager_ids
 *?
_output_shapest
r:?????????:?????????:?????????:?????????:?????????:?????????*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_105422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*?
_input_shapes
}:???????????::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_4
?
`
D__inference_flatten_3_layer_call_and_return_conditional_losses_11512

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
C__inference_dense_34_layer_call_and_return_conditional_losses_11654

inputs)
%matmul_readvariableop_dense_34_kernel(
$biasadd_readvariableop_dense_34_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_34_kernel*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_34_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?e
?

B__inference_model_3_layer_call_and_return_conditional_losses_10960
input_4
conv2d_11_conv2d_11_kernel
conv2d_11_conv2d_11_bias
conv2d_12_conv2d_12_kernel
conv2d_12_conv2d_12_bias
conv2d_13_conv2d_13_kernel
conv2d_13_conv2d_13_bias
dense_27_dense_27_kernel
dense_27_dense_27_bias
dense_28_dense_28_kernel
dense_28_dense_28_bias
dense_29_dense_29_kernel
dense_29_dense_29_bias
dense_35_dense_35_kernel
dense_35_dense_35_bias
dense_34_dense_34_kernel
dense_34_dense_34_bias
dense_33_dense_33_kernel
dense_33_dense_33_bias
dense_32_dense_32_kernel
dense_32_dense_32_bias
dense_31_dense_31_kernel
dense_31_dense_31_bias
dense_30_dense_30_kernel
dense_30_dense_30_bias
identity

identity_1

identity_2

identity_3

identity_4

identity_5??!conv2d_11/StatefulPartitionedCall?!conv2d_12/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall? dense_27/StatefulPartitionedCall? dense_28/StatefulPartitionedCall? dense_29/StatefulPartitionedCall? dense_30/StatefulPartitionedCall? dense_31/StatefulPartitionedCall? dense_32/StatefulPartitionedCall? dense_33/StatefulPartitionedCall? dense_34/StatefulPartitionedCall? dense_35/StatefulPartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCallinput_4conv2d_11_conv2d_11_kernelconv2d_11_conv2d_11_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_106082#
!conv2d_11/StatefulPartitionedCall?
 max_pooling2d_10/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_105562"
 max_pooling2d_10/PartitionedCall?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_10/PartitionedCall:output:0conv2d_12_conv2d_12_kernelconv2d_12_conv2d_12_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(( *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_106372#
!conv2d_12/StatefulPartitionedCall?
 max_pooling2d_11/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_105732"
 max_pooling2d_11/PartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_11/PartitionedCall:output:0conv2d_13_conv2d_13_kernelconv2d_13_conv2d_13_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_106662#
!conv2d_13/StatefulPartitionedCall?
 max_pooling2d_12/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_105902"
 max_pooling2d_12/PartitionedCall?
flatten_3/PartitionedCallPartitionedCall)max_pooling2d_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_106902
flatten_3/PartitionedCall?
 dense_27/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_27_dense_27_kerneldense_27_dense_27_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_27_layer_call_and_return_conditional_losses_107092"
 dense_27/StatefulPartitionedCall?
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0dense_28_dense_28_kerneldense_28_dense_28_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_28_layer_call_and_return_conditional_losses_107322"
 dense_28/StatefulPartitionedCall?
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_dense_29_kerneldense_29_dense_29_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_29_layer_call_and_return_conditional_losses_107552"
 dense_29/StatefulPartitionedCall?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_29/StatefulPartitionedCall:output:0dense_35_dense_35_kerneldense_35_dense_35_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_35_layer_call_and_return_conditional_losses_107782"
 dense_35/StatefulPartitionedCall?
 dense_34/StatefulPartitionedCallStatefulPartitionedCall)dense_29/StatefulPartitionedCall:output:0dense_34_dense_34_kerneldense_34_dense_34_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_34_layer_call_and_return_conditional_losses_108012"
 dense_34/StatefulPartitionedCall?
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_29/StatefulPartitionedCall:output:0dense_33_dense_33_kerneldense_33_dense_33_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_33_layer_call_and_return_conditional_losses_108242"
 dense_33/StatefulPartitionedCall?
 dense_32/StatefulPartitionedCallStatefulPartitionedCall)dense_29/StatefulPartitionedCall:output:0dense_32_dense_32_kerneldense_32_dense_32_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_32_layer_call_and_return_conditional_losses_108472"
 dense_32/StatefulPartitionedCall?
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_29/StatefulPartitionedCall:output:0dense_31_dense_31_kerneldense_31_dense_31_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_31_layer_call_and_return_conditional_losses_108702"
 dense_31/StatefulPartitionedCall?
 dense_30/StatefulPartitionedCallStatefulPartitionedCall)dense_29/StatefulPartitionedCall:output:0dense_30_dense_30_kerneldense_30_dense_30_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_30_layer_call_and_return_conditional_losses_108932"
 dense_30/StatefulPartitionedCall?
IdentityIdentity)dense_30/StatefulPartitionedCall:output:0"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity)dense_31/StatefulPartitionedCall:output:0"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity)dense_32/StatefulPartitionedCall:output:0"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity)dense_33/StatefulPartitionedCall:output:0"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity)dense_34/StatefulPartitionedCall:output:0"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity)dense_35/StatefulPartitionedCall:output:0"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*?
_input_shapes
}:???????????::::::::::::::::::::::::2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_4
?
?
(__inference_dense_33_layer_call_fn_11643

inputs
dense_33_kernel
dense_33_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_33_kerneldense_33_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_33_layer_call_and_return_conditional_losses_108242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
C__inference_dense_34_layer_call_and_return_conditional_losses_10801

inputs)
%matmul_readvariableop_dense_34_kernel(
$biasadd_readvariableop_dense_34_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_34_kernel*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_34_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
'__inference_model_3_layer_call_fn_11452

inputs
conv2d_11_kernel
conv2d_11_bias
conv2d_12_kernel
conv2d_12_bias
conv2d_13_kernel
conv2d_13_bias
dense_27_kernel
dense_27_bias
dense_28_kernel
dense_28_bias
dense_29_kernel
dense_29_bias
dense_35_kernel
dense_35_bias
dense_34_kernel
dense_34_bias
dense_33_kernel
dense_33_bias
dense_32_kernel
dense_32_bias
dense_31_kernel
dense_31_bias
dense_30_kernel
dense_30_bias
identity

identity_1

identity_2

identity_3

identity_4

identity_5??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_11_kernelconv2d_11_biasconv2d_12_kernelconv2d_12_biasconv2d_13_kernelconv2d_13_biasdense_27_kerneldense_27_biasdense_28_kerneldense_28_biasdense_29_kerneldense_29_biasdense_35_kerneldense_35_biasdense_34_kerneldense_34_biasdense_33_kerneldense_33_biasdense_32_kerneldense_32_biasdense_31_kerneldense_31_biasdense_30_kerneldense_30_bias*$
Tin
2*
Tout

2*
_collective_manager_ids
 *?
_output_shapest
r:?????????:?????????:?????????:?????????:?????????:?????????*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_111002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*?
_input_shapes
}:???????????::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
(__inference_dense_32_layer_call_fn_11625

inputs
dense_32_kernel
dense_32_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_32_kerneldense_32_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_32_layer_call_and_return_conditional_losses_108472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
D__inference_conv2d_12_layer_call_and_return_conditional_losses_11481

inputs*
&conv2d_readvariableop_conv2d_12_kernel)
%biasadd_readvariableop_conv2d_12_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp&conv2d_readvariableop_conv2d_12_kernel*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(( *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_conv2d_12_bias*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(( 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????(( 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????(( 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????**::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????**
 
_user_specified_nameinputs
?
?
(__inference_dense_34_layer_call_fn_11661

inputs
dense_34_kernel
dense_34_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_34_kerneldense_34_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_34_layer_call_and_return_conditional_losses_108012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
C__inference_dense_29_layer_call_and_return_conditional_losses_10755

inputs)
%matmul_readvariableop_dense_29_kernel(
$biasadd_readvariableop_dense_29_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_29_kernel*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_29_bias*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_10548

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
(__inference_dense_31_layer_call_fn_11607

inputs
dense_31_kernel
dense_31_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_31_kerneldense_31_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_31_layer_call_and_return_conditional_losses_108702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
C__inference_dense_33_layer_call_and_return_conditional_losses_10824

inputs)
%matmul_readvariableop_dense_33_kernel(
$biasadd_readvariableop_dense_33_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_33_kernel*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_33_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_12_layer_call_fn_10593

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_105902
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
(__inference_dense_27_layer_call_fn_11535

inputs
dense_27_kernel
dense_27_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_27_kerneldense_27_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_27_layer_call_and_return_conditional_losses_107092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
input_4:
serving_default_input_4:0???????????<
dense_300
StatefulPartitionedCall:0?????????<
dense_310
StatefulPartitionedCall:1?????????<
dense_320
StatefulPartitionedCall:2?????????<
dense_330
StatefulPartitionedCall:3?????????<
dense_340
StatefulPartitionedCall:4?????????<
dense_350
StatefulPartitionedCall:5?????????tensorflow/serving/predict:??
??
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer_with_weights-8
layer-13
layer_with_weights-9
layer-14
layer_with_weights-10
layer-15
layer_with_weights-11
layer-16
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"Ë
_tf_keras_network??{"class_name": "Functional", "name": "model_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_11", "inbound_nodes": [[["input_4", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_10", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_10", "inbound_nodes": [[["conv2d_11", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_12", "inbound_nodes": [[["max_pooling2d_10", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_11", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_11", "inbound_nodes": [[["conv2d_12", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_13", "inbound_nodes": [[["max_pooling2d_11", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_12", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_12", "inbound_nodes": [[["conv2d_13", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_3", "inbound_nodes": [[["max_pooling2d_12", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_27", "inbound_nodes": [[["flatten_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_28", "inbound_nodes": [[["dense_27", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_29", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_29", "inbound_nodes": [[["dense_28", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_30", "trainable": true, "dtype": "float32", "units": 16, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_30", "inbound_nodes": [[["dense_29", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_31", "trainable": true, "dtype": "float32", "units": 16, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_31", "inbound_nodes": [[["dense_29", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_32", "trainable": true, "dtype": "float32", "units": 16, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_32", "inbound_nodes": [[["dense_29", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 16, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_33", "inbound_nodes": [[["dense_29", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 16, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_34", "inbound_nodes": [[["dense_29", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 16, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_35", "inbound_nodes": [[["dense_29", 0, 0, {}]]]}], "input_layers": [["input_4", 0, 0]], "output_layers": [["dense_30", 0, 0], ["dense_31", 0, 0], ["dense_32", 0, 0], ["dense_33", 0, 0], ["dense_34", 0, 0], ["dense_35", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_11", "inbound_nodes": [[["input_4", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_10", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_10", "inbound_nodes": [[["conv2d_11", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_12", "inbound_nodes": [[["max_pooling2d_10", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_11", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_11", "inbound_nodes": [[["conv2d_12", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_13", "inbound_nodes": [[["max_pooling2d_11", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_12", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_12", "inbound_nodes": [[["conv2d_13", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_3", "inbound_nodes": [[["max_pooling2d_12", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_27", "inbound_nodes": [[["flatten_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_28", "inbound_nodes": [[["dense_27", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_29", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_29", "inbound_nodes": [[["dense_28", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_30", "trainable": true, "dtype": "float32", "units": 16, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_30", "inbound_nodes": [[["dense_29", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_31", "trainable": true, "dtype": "float32", "units": 16, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_31", "inbound_nodes": [[["dense_29", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_32", "trainable": true, "dtype": "float32", "units": 16, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_32", "inbound_nodes": [[["dense_29", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 16, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_33", "inbound_nodes": [[["dense_29", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 16, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_34", "inbound_nodes": [[["dense_29", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 16, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_35", "inbound_nodes": [[["dense_29", 0, 0, {}]]]}], "input_layers": [["input_4", 0, 0]], "output_layers": [["dense_30", 0, 0], ["dense_31", 0, 0], ["dense_32", 0, 0], ["dense_33", 0, 0], ["dense_34", 0, 0], ["dense_35", 0, 0]]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": true}}, "metrics": ["accuracy"], "loss_weights": null, "sample_weight_mode": null, "weighted_metrics": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.5, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_4", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}}
?	

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 1]}}
?
trainable_variables
regularization_losses
 	variables
!	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_10", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

"kernel
#bias
$trainable_variables
%regularization_losses
&	variables
'	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 42, 42, 3]}}
?
(trainable_variables
)regularization_losses
*	variables
+	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_11", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

,kernel
-bias
.trainable_variables
/regularization_losses
0	variables
1	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 13, 13, 32]}}
?
2trainable_variables
3regularization_losses
4	variables
5	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_12", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
6trainable_variables
7regularization_losses
8	variables
9	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

:kernel
;bias
<trainable_variables
=regularization_losses
>	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1600}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1600]}}
?

@kernel
Abias
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?

Fkernel
Gbias
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_29", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_29", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?

Lkernel
Mbias
Ntrainable_variables
Oregularization_losses
P	variables
Q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_30", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_30", "trainable": true, "dtype": "float32", "units": 16, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?

Rkernel
Sbias
Ttrainable_variables
Uregularization_losses
V	variables
W	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_31", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_31", "trainable": true, "dtype": "float32", "units": 16, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?

Xkernel
Ybias
Ztrainable_variables
[regularization_losses
\	variables
]	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_32", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_32", "trainable": true, "dtype": "float32", "units": 16, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?

^kernel
_bias
`trainable_variables
aregularization_losses
b	variables
c	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_33", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 16, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?

dkernel
ebias
ftrainable_variables
gregularization_losses
h	variables
i	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 16, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?

jkernel
kbias
ltrainable_variables
mregularization_losses
n	variables
o	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 16, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?
piter

qbeta_1

rbeta_2
	sdecay
tlearning_ratem?m?"m?#m?,m?-m?:m?;m?@m?Am?Fm?Gm?Lm?Mm?Rm?Sm?Xm?Ym?^m?_m?dm?em?jm?km?v?v?"v?#v?,v?-v?:v?;v?@v?Av?Fv?Gv?Lv?Mv?Rv?Sv?Xv?Yv?^v?_v?dv?ev?jv?kv?"
	optimizer
?
0
1
"2
#3
,4
-5
:6
;7
@8
A9
F10
G11
L12
M13
R14
S15
X16
Y17
^18
_19
d20
e21
j22
k23"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
"2
#3
,4
-5
:6
;7
@8
A9
F10
G11
L12
M13
R14
S15
X16
Y17
^18
_19
d20
e21
j22
k23"
trackable_list_wrapper
?
ulayer_metrics
trainable_variables
vnon_trainable_variables
wmetrics
regularization_losses
xlayer_regularization_losses
	variables

ylayers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
*:(2conv2d_11/kernel
:2conv2d_11/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
zlayer_metrics
trainable_variables
{non_trainable_variables
|metrics
regularization_losses
}layer_regularization_losses
	variables

~layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
layer_metrics
trainable_variables
?non_trainable_variables
?metrics
regularization_losses
 ?layer_regularization_losses
 	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( 2conv2d_12/kernel
: 2conv2d_12/bias
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
?
?layer_metrics
$trainable_variables
?non_trainable_variables
?metrics
%regularization_losses
 ?layer_regularization_losses
&	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
(trainable_variables
?non_trainable_variables
?metrics
)regularization_losses
 ?layer_regularization_losses
*	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( @2conv2d_13/kernel
:@2conv2d_13/bias
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
?
?layer_metrics
.trainable_variables
?non_trainable_variables
?metrics
/regularization_losses
 ?layer_regularization_losses
0	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
2trainable_variables
?non_trainable_variables
?metrics
3regularization_losses
 ?layer_regularization_losses
4	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
6trainable_variables
?non_trainable_variables
?metrics
7regularization_losses
 ?layer_regularization_losses
8	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_27/kernel
:?2dense_27/bias
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
?
?layer_metrics
<trainable_variables
?non_trainable_variables
?metrics
=regularization_losses
 ?layer_regularization_losses
>	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?@2dense_28/kernel
:@2dense_28/bias
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
?
?layer_metrics
Btrainable_variables
?non_trainable_variables
?metrics
Cregularization_losses
 ?layer_regularization_losses
D	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:@ 2dense_29/kernel
: 2dense_29/bias
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
?
?layer_metrics
Htrainable_variables
?non_trainable_variables
?metrics
Iregularization_losses
 ?layer_regularization_losses
J	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!: 2dense_30/kernel
:2dense_30/bias
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
?
?layer_metrics
Ntrainable_variables
?non_trainable_variables
?metrics
Oregularization_losses
 ?layer_regularization_losses
P	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!: 2dense_31/kernel
:2dense_31/bias
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
?
?layer_metrics
Ttrainable_variables
?non_trainable_variables
?metrics
Uregularization_losses
 ?layer_regularization_losses
V	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!: 2dense_32/kernel
:2dense_32/bias
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
?
?layer_metrics
Ztrainable_variables
?non_trainable_variables
?metrics
[regularization_losses
 ?layer_regularization_losses
\	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!: 2dense_33/kernel
:2dense_33/bias
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
?
?layer_metrics
`trainable_variables
?non_trainable_variables
?metrics
aregularization_losses
 ?layer_regularization_losses
b	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!: 2dense_34/kernel
:2dense_34/bias
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
?
?layer_metrics
ftrainable_variables
?non_trainable_variables
?metrics
gregularization_losses
 ?layer_regularization_losses
h	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!: 2dense_35/kernel
:2dense_35/bias
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
?
?layer_metrics
ltrainable_variables
?non_trainable_variables
?metrics
mregularization_losses
 ?layer_regularization_losses
n	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2training_8/Adam/iter
 : (2training_8/Adam/beta_1
 : (2training_8/Adam/beta_2
: (2training_8/Adam/decay
':% (2training_8/Adam/learning_rate
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
P
?0
?1
?2
?3
?4
?5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "dense_30_accuracy", "dtype": "float32", "config": {"name": "dense_30_accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "dense_31_accuracy", "dtype": "float32", "config": {"name": "dense_31_accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "dense_32_accuracy", "dtype": "float32", "config": {"name": "dense_32_accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "dense_33_accuracy", "dtype": "float32", "config": {"name": "dense_33_accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "dense_34_accuracy", "dtype": "float32", "config": {"name": "dense_34_accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "dense_35_accuracy", "dtype": "float32", "config": {"name": "dense_35_accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
:  (2total_48
:  (2count_48
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total_49
:  (2count_49
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total_50
:  (2count_50
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total_51
:  (2count_51
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total_52
:  (2count_52
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total_53
:  (2count_53
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
::82"training_8/Adam/conv2d_11/kernel/m
,:*2 training_8/Adam/conv2d_11/bias/m
::8 2"training_8/Adam/conv2d_12/kernel/m
,:* 2 training_8/Adam/conv2d_12/bias/m
::8 @2"training_8/Adam/conv2d_13/kernel/m
,:*@2 training_8/Adam/conv2d_13/bias/m
3:1
??2!training_8/Adam/dense_27/kernel/m
,:*?2training_8/Adam/dense_27/bias/m
2:0	?@2!training_8/Adam/dense_28/kernel/m
+:)@2training_8/Adam/dense_28/bias/m
1:/@ 2!training_8/Adam/dense_29/kernel/m
+:) 2training_8/Adam/dense_29/bias/m
1:/ 2!training_8/Adam/dense_30/kernel/m
+:)2training_8/Adam/dense_30/bias/m
1:/ 2!training_8/Adam/dense_31/kernel/m
+:)2training_8/Adam/dense_31/bias/m
1:/ 2!training_8/Adam/dense_32/kernel/m
+:)2training_8/Adam/dense_32/bias/m
1:/ 2!training_8/Adam/dense_33/kernel/m
+:)2training_8/Adam/dense_33/bias/m
1:/ 2!training_8/Adam/dense_34/kernel/m
+:)2training_8/Adam/dense_34/bias/m
1:/ 2!training_8/Adam/dense_35/kernel/m
+:)2training_8/Adam/dense_35/bias/m
::82"training_8/Adam/conv2d_11/kernel/v
,:*2 training_8/Adam/conv2d_11/bias/v
::8 2"training_8/Adam/conv2d_12/kernel/v
,:* 2 training_8/Adam/conv2d_12/bias/v
::8 @2"training_8/Adam/conv2d_13/kernel/v
,:*@2 training_8/Adam/conv2d_13/bias/v
3:1
??2!training_8/Adam/dense_27/kernel/v
,:*?2training_8/Adam/dense_27/bias/v
2:0	?@2!training_8/Adam/dense_28/kernel/v
+:)@2training_8/Adam/dense_28/bias/v
1:/@ 2!training_8/Adam/dense_29/kernel/v
+:) 2training_8/Adam/dense_29/bias/v
1:/ 2!training_8/Adam/dense_30/kernel/v
+:)2training_8/Adam/dense_30/bias/v
1:/ 2!training_8/Adam/dense_31/kernel/v
+:)2training_8/Adam/dense_31/bias/v
1:/ 2!training_8/Adam/dense_32/kernel/v
+:)2training_8/Adam/dense_32/bias/v
1:/ 2!training_8/Adam/dense_33/kernel/v
+:)2training_8/Adam/dense_33/bias/v
1:/ 2!training_8/Adam/dense_34/kernel/v
+:)2training_8/Adam/dense_34/bias/v
1:/ 2!training_8/Adam/dense_35/kernel/v
+:)2training_8/Adam/dense_35/bias/v
?2?
 __inference__wrapped_model_10542?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *0?-
+?(
input_4???????????
?2?
'__inference_model_3_layer_call_fn_11413
'__inference_model_3_layer_call_fn_11137
'__inference_model_3_layer_call_fn_11049
'__inference_model_3_layer_call_fn_11452?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_model_3_layer_call_and_return_conditional_losses_10960
B__inference_model_3_layer_call_and_return_conditional_losses_10911
B__inference_model_3_layer_call_and_return_conditional_losses_11276
B__inference_model_3_layer_call_and_return_conditional_losses_11374?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_conv2d_11_layer_call_fn_11470?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_11_layer_call_and_return_conditional_losses_11463?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_max_pooling2d_10_layer_call_fn_10559?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_10548?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
)__inference_conv2d_12_layer_call_fn_11488?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_12_layer_call_and_return_conditional_losses_11481?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_max_pooling2d_11_layer_call_fn_10576?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_10565?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
)__inference_conv2d_13_layer_call_fn_11506?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_13_layer_call_and_return_conditional_losses_11499?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_max_pooling2d_12_layer_call_fn_10593?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_10582?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
)__inference_flatten_3_layer_call_fn_11517?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_3_layer_call_and_return_conditional_losses_11512?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_27_layer_call_fn_11535?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_27_layer_call_and_return_conditional_losses_11528?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_28_layer_call_fn_11553?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_28_layer_call_and_return_conditional_losses_11546?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_29_layer_call_fn_11571?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_29_layer_call_and_return_conditional_losses_11564?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_30_layer_call_fn_11589?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_30_layer_call_and_return_conditional_losses_11582?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_31_layer_call_fn_11607?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_31_layer_call_and_return_conditional_losses_11600?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_32_layer_call_fn_11625?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_32_layer_call_and_return_conditional_losses_11618?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_33_layer_call_fn_11643?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_33_layer_call_and_return_conditional_losses_11636?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_34_layer_call_fn_11661?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_34_layer_call_and_return_conditional_losses_11654?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_35_layer_call_fn_11679?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_35_layer_call_and_return_conditional_losses_11672?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_11178input_4"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_10542?"#,-:;@AFGjkde^_XYRSLM:?7
0?-
+?(
input_4???????????
? "???
.
dense_30"?
dense_30?????????
.
dense_31"?
dense_31?????????
.
dense_32"?
dense_32?????????
.
dense_33"?
dense_33?????????
.
dense_34"?
dense_34?????????
.
dense_35"?
dense_35??????????
D__inference_conv2d_11_layer_call_and_return_conditional_losses_11463p9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
)__inference_conv2d_11_layer_call_fn_11470c9?6
/?,
*?'
inputs???????????
? ""?????????????
D__inference_conv2d_12_layer_call_and_return_conditional_losses_11481l"#7?4
-?*
(?%
inputs?????????**
? "-?*
#? 
0?????????(( 
? ?
)__inference_conv2d_12_layer_call_fn_11488_"#7?4
-?*
(?%
inputs?????????**
? " ??????????(( ?
D__inference_conv2d_13_layer_call_and_return_conditional_losses_11499l,-7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????@
? ?
)__inference_conv2d_13_layer_call_fn_11506_,-7?4
-?*
(?%
inputs????????? 
? " ??????????@?
C__inference_dense_27_layer_call_and_return_conditional_losses_11528^:;0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? }
(__inference_dense_27_layer_call_fn_11535Q:;0?-
&?#
!?
inputs??????????
? "????????????
C__inference_dense_28_layer_call_and_return_conditional_losses_11546]@A0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? |
(__inference_dense_28_layer_call_fn_11553P@A0?-
&?#
!?
inputs??????????
? "??????????@?
C__inference_dense_29_layer_call_and_return_conditional_losses_11564\FG/?,
%?"
 ?
inputs?????????@
? "%?"
?
0????????? 
? {
(__inference_dense_29_layer_call_fn_11571OFG/?,
%?"
 ?
inputs?????????@
? "?????????? ?
C__inference_dense_30_layer_call_and_return_conditional_losses_11582\LM/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? {
(__inference_dense_30_layer_call_fn_11589OLM/?,
%?"
 ?
inputs????????? 
? "???????????
C__inference_dense_31_layer_call_and_return_conditional_losses_11600\RS/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? {
(__inference_dense_31_layer_call_fn_11607ORS/?,
%?"
 ?
inputs????????? 
? "???????????
C__inference_dense_32_layer_call_and_return_conditional_losses_11618\XY/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? {
(__inference_dense_32_layer_call_fn_11625OXY/?,
%?"
 ?
inputs????????? 
? "???????????
C__inference_dense_33_layer_call_and_return_conditional_losses_11636\^_/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? {
(__inference_dense_33_layer_call_fn_11643O^_/?,
%?"
 ?
inputs????????? 
? "???????????
C__inference_dense_34_layer_call_and_return_conditional_losses_11654\de/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? {
(__inference_dense_34_layer_call_fn_11661Ode/?,
%?"
 ?
inputs????????? 
? "???????????
C__inference_dense_35_layer_call_and_return_conditional_losses_11672\jk/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? {
(__inference_dense_35_layer_call_fn_11679Ojk/?,
%?"
 ?
inputs????????? 
? "???????????
D__inference_flatten_3_layer_call_and_return_conditional_losses_11512a7?4
-?*
(?%
inputs?????????@
? "&?#
?
0??????????
? ?
)__inference_flatten_3_layer_call_fn_11517T7?4
-?*
(?%
inputs?????????@
? "????????????
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_10548?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_max_pooling2d_10_layer_call_fn_10559?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_10565?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_max_pooling2d_11_layer_call_fn_10576?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_10582?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_max_pooling2d_12_layer_call_fn_10593?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
B__inference_model_3_layer_call_and_return_conditional_losses_10911?"#,-:;@AFGjkde^_XYRSLMB??
8?5
+?(
input_4???????????
p

 
? "???
???
?
0/0?????????
?
0/1?????????
?
0/2?????????
?
0/3?????????
?
0/4?????????
?
0/5?????????
? ?
B__inference_model_3_layer_call_and_return_conditional_losses_10960?"#,-:;@AFGjkde^_XYRSLMB??
8?5
+?(
input_4???????????
p 

 
? "???
???
?
0/0?????????
?
0/1?????????
?
0/2?????????
?
0/3?????????
?
0/4?????????
?
0/5?????????
? ?
B__inference_model_3_layer_call_and_return_conditional_losses_11276?"#,-:;@AFGjkde^_XYRSLMA?>
7?4
*?'
inputs???????????
p

 
? "???
???
?
0/0?????????
?
0/1?????????
?
0/2?????????
?
0/3?????????
?
0/4?????????
?
0/5?????????
? ?
B__inference_model_3_layer_call_and_return_conditional_losses_11374?"#,-:;@AFGjkde^_XYRSLMA?>
7?4
*?'
inputs???????????
p 

 
? "???
???
?
0/0?????????
?
0/1?????????
?
0/2?????????
?
0/3?????????
?
0/4?????????
?
0/5?????????
? ?
'__inference_model_3_layer_call_fn_11049?"#,-:;@AFGjkde^_XYRSLMB??
8?5
+?(
input_4???????????
p

 
? "???
?
0?????????
?
1?????????
?
2?????????
?
3?????????
?
4?????????
?
5??????????
'__inference_model_3_layer_call_fn_11137?"#,-:;@AFGjkde^_XYRSLMB??
8?5
+?(
input_4???????????
p 

 
? "???
?
0?????????
?
1?????????
?
2?????????
?
3?????????
?
4?????????
?
5??????????
'__inference_model_3_layer_call_fn_11413?"#,-:;@AFGjkde^_XYRSLMA?>
7?4
*?'
inputs???????????
p

 
? "???
?
0?????????
?
1?????????
?
2?????????
?
3?????????
?
4?????????
?
5??????????
'__inference_model_3_layer_call_fn_11452?"#,-:;@AFGjkde^_XYRSLMA?>
7?4
*?'
inputs???????????
p 

 
? "???
?
0?????????
?
1?????????
?
2?????????
?
3?????????
?
4?????????
?
5??????????
#__inference_signature_wrapper_11178?"#,-:;@AFGjkde^_XYRSLME?B
? 
;?8
6
input_4+?(
input_4???????????"???
.
dense_30"?
dense_30?????????
.
dense_31"?
dense_31?????????
.
dense_32"?
dense_32?????????
.
dense_33"?
dense_33?????????
.
dense_34"?
dense_34?????????
.
dense_35"?
dense_35?????????