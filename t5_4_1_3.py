# -*- coding: utf-8 -*-
"""
@author: tz_zs
单文件 常量的方式保存115
convert_variables_to_constants
"""

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

v1 = tf.Variable(1, dtype=tf.float32, name="v1")
v2 = tf.Variable(2, dtype=tf.float32, name="v2")
result = v1 + v2

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # print(tf.get_default_graph())  #<tensorflow.python.framework.ops.Graph object at 0x0000020329B56C88>
    graph_def = tf.get_default_graph().as_graph_def()
    print("#####" * 5 + " 得到当前默认的计算图graph_def " + "#####" * 5)
    print(graph_def)
    '''
    node {
      name: "v1/initial_value"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_FLOAT
            tensor_shape {
            }
            float_val: 1.0
          }
        }
      }
    }
    node {
      name: "v1"
      op: "VariableV2"
      attr {
        key: "container"
        value {
          s: ""
        }
      }
      attr {
        key: "dtype"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "shape"
        value {
          shape {
          }
        }
      }
      attr {
        key: "shared_name"
        value {
          s: ""
        }
      }
    }
    node {
      name: "v1/Assign"
      op: "Assign"
      input: "v1"
      input: "v1/initial_value"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "_class"
        value {
          list {
            s: "loc:@v1"
          }
        }
      }
      attr {
        key: "use_locking"
        value {
          b: true
        }
      }
      attr {
        key: "validate_shape"
        value {
          b: true
        }
      }
    }
    node {
      name: "v1/read"
      op: "Identity"
      input: "v1"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "_class"
        value {
          list {
            s: "loc:@v1"
          }
        }
      }
    }
    node {
      name: "v2/initial_value"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_FLOAT
            tensor_shape {
            }
            float_val: 2.0
          }
        }
      }
    }
    node {
      name: "v2"
      op: "VariableV2"
      attr {
        key: "container"
        value {
          s: ""
        }
      }
      attr {
        key: "dtype"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "shape"
        value {
          shape {
          }
        }
      }
      attr {
        key: "shared_name"
        value {
          s: ""
        }
      }
    }
    node {
      name: "v2/Assign"
      op: "Assign"
      input: "v2"
      input: "v2/initial_value"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "_class"
        value {
          list {
            s: "loc:@v2"
          }
        }
      }
      attr {
        key: "use_locking"
        value {
          b: true
        }
      }
      attr {
        key: "validate_shape"
        value {
          b: true
        }
      }
    }
    node {
      name: "v2/read"
      op: "Identity"
      input: "v2"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "_class"
        value {
          list {
            s: "loc:@v2"
          }
        }
      }
    }
    node {
      name: "add"
      op: "Add"
      input: "v1/read"
      input: "v2/read"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
    }
    node {
      name: "init"
      op: "NoOp"
      input: "^v1/Assign"
      input: "^v2/Assign"
    }
    versions {
      producer: 24
    }
    '''

    print("#####" * 5 + " add节点相关的output_graph_def " + "#####" * 5)
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ["add"])
    print(output_graph_def)
    '''
    Converted 2 variables to const ops.
    node {
      name: "v1"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_FLOAT
            tensor_shape {
            }
            float_val: 1.0
          }
        }
      }
    }
    node {
      name: "v1/read"
      op: "Identity"
      input: "v1"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "_class"
        value {
          list {
            s: "loc:@v1"
          }
        }
      }
    }
    node {
      name: "v2"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_FLOAT
            tensor_shape {
            }
            float_val: 2.0
          }
        }
      }
    }
    node {
      name: "v2/read"
      op: "Identity"
      input: "v2"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "_class"
        value {
          list {
            s: "loc:@v2"
          }
        }
      }
    }
    node {
      name: "add"
      op: "Add"
      input: "v1/read"
      input: "v2/read"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
    }
    library {
    }
    '''
    with tf.gfile.GFile("D:/path/to/model115/combined_model.pb", "wb") as f:
        f.write(output_graph_def.SerializeToString())

print("#####" * 5 + " 加载模型 " + "#####" * 5)
with tf.Session() as sess:
    model_filename = "D:/path/to/model115/combined_model.pb"
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def2 = tf.GraphDef()
        graph_def2.ParseFromString(f.read())
        print(graph_def2)
        '''
        node {
          name: "v1"
          op: "Const"
          attr {
            key: "dtype"
            value {
              type: DT_FLOAT
            }
          }
          attr {
            key: "value"
            value {
              tensor {
                dtype: DT_FLOAT
                tensor_shape {
                }
                float_val: 1.0
              }
            }
          }
        }
        node {
          name: "v1/read"
          op: "Identity"
          input: "v1"
          attr {
            key: "T"
            value {
              type: DT_FLOAT
            }
          }
          attr {
            key: "_class"
            value {
              list {
                s: "loc:@v1"
              }
            }
          }
        }
        node {
          name: "v2"
          op: "Const"
          attr {
            key: "dtype"
            value {
              type: DT_FLOAT
            }
          }
          attr {
            key: "value"
            value {
              tensor {
                dtype: DT_FLOAT
                tensor_shape {
                }
                float_val: 2.0
              }
            }
          }
        }
        node {
          name: "v2/read"
          op: "Identity"
          input: "v2"
          attr {
            key: "T"
            value {
              type: DT_FLOAT
            }
          }
          attr {
            key: "_class"
            value {
              list {
                s: "loc:@v2"
              }
            }
          }
        }
        node {
          name: "add"
          op: "Add"
          input: "v1/read"
          input: "v2/read"
          attr {
            key: "T"
            value {
              type: DT_FLOAT
            }
          }
        }
        library {
        }
        '''

    print("#" * 10)
    # 可以通过张量的名称取某个张量的值
    result = tf.import_graph_def(graph_def2, return_elements=["add:0"])
    result_v1 = tf.import_graph_def(graph_def2, return_elements=["v1:0"])
    print(result)  # [<tf.Tensor 'import/add:0' shape=() dtype=float32>]
    print(sess.run(result))  # [3.0]
    print(result_v1)  # [<tf.Tensor 'import_1/v1:0' shape=() dtype=float32>]
    print(sess.run(result_v1))  # [1.0]
