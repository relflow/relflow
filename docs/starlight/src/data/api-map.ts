/** Navigation only. Reference content is extracted from the checked-out Python source. */
export interface ApiEntry {
  name: string;
  guide?: string;
  extra?: string;
  upstream?: string;
}

export interface ApiSection {
  id: string;
  title: string;
  description: string;
  entries: ApiEntry[];
}

export const apiCatalog: ApiSection[] = [
  {
    "id": "schema",
    "title": "Build a model",
    "description": "Declare records, choose architecture defaults, and configure how context reaches prediction targets.",
    "entries": [
      {
        "name": "rf.Model",
        "guide": "/concepts/models/"
      },
      {
        "name": "rf.Branch",
        "guide": "/concepts/branches/"
      },
      {
        "name": "rf.Schema"
      },
      {
        "name": "rf.Model.from_yaml"
      },
      {
        "name": "rf.Mask",
        "guide": "/concepts/targets/"
      },
      {
        "name": "rf.Jitter"
      },
      {
        "name": "rf.Attention"
      },
      {
        "name": "rf.Mean"
      },
      {
        "name": "rf.presets",
        "guide": "/concepts/runtime/"
      },
      {
        "name": "rf.presets.SMALL",
        "guide": "/concepts/runtime/"
      },
      {
        "name": "rf.presets.MEDIUM",
        "guide": "/concepts/runtime/"
      },
      {
        "name": "rf.presets.get"
      },
      {
        "name": "rf.presets.register"
      },
      {
        "name": "rf.presets.Preset"
      },
      {
        "name": "rf.presets.BranchDefaults"
      },
      {
        "name": "rf.presets.LeafDefaults"
      },
      {
        "name": "rf.presets.FitDefaults"
      }
    ]
  },
  {
    "id": "types",
    "title": "Choose a representation",
    "description": "Give each value a meaning that determines its encoding, objectives, and decoded output contract.",
    "entries": [
      {
        "name": "rf.Number",
        "guide": "/data-types/number/"
      },
      {
        "name": "rf.Quantile",
        "guide": "/data-types/quantile/"
      },
      {
        "name": "rf.Ordinal",
        "guide": "/data-types/ordinal/"
      },
      {
        "name": "rf.Boolean",
        "guide": "/data-types/boolean/"
      },
      {
        "name": "rf.Category",
        "guide": "/data-types/category/"
      },
      {
        "name": "rf.Enum",
        "guide": "/data-types/enum/"
      },
      {
        "name": "rf.Set",
        "guide": "/data-types/set/"
      },
      {
        "name": "rf.Hash",
        "guide": "/data-types/hash/"
      },
      {
        "name": "rf.Group",
        "guide": "/data-types/group/"
      },
      {
        "name": "rf.Period",
        "guide": "/data-types/period/"
      },
      {
        "name": "rf.Duration",
        "guide": "/data-types/duration/"
      },
      {
        "name": "rf.Text",
        "guide": "/data-types/text/"
      },
      {
        "name": "rf.Vector",
        "guide": "/data-types/vector/"
      }
    ]
  },
  {
    "id": "data",
    "title": "Load and prepare data",
    "description": "Supply named data splits and turn source records into the observations described by your schema.",
    "entries": [
      {
        "name": "rf.ArrowDataModule",
        "guide": "/advanced/data-modules/"
      },
      {
        "name": "rf.PolarsDataModule"
      },
      {
        "name": "rf.DuckDBDataModule"
      },
      {
        "name": "rf.CustomDataModule",
        "guide": "/advanced/custom-data-modules/"
      },
      {
        "name": "rf.SyntheticDataModule"
      },
      {
        "name": "rf.source"
      },
      {
        "name": "rf.preprocess",
        "guide": "/advanced/data-modules/"
      },
      {
        "name": "rf.Preprocessor"
      }
    ]
  },
  {
    "id": "inspection",
    "title": "Inspect artifacts and data",
    "description": "Check the declared learning contract and its input sources before training or deployment.",
    "entries": [
      {
        "name": "rf.inspect",
        "guide": "/advanced/command-line/"
      },
      {
        "name": "rf.check",
        "guide": "/advanced/command-line/"
      }
    ]
  },
  {
    "id": "training",
    "title": "Train and evaluate",
    "description": "Fit a model, evaluate held-out data, and configure optimization or selected runtime compilation.",
    "entries": [
      {
        "name": "model.fit",
        "guide": "/concepts/runtime/"
      },
      {
        "name": "model.validate",
        "guide": "/concepts/runtime/"
      },
      {
        "name": "model.test",
        "guide": "/concepts/runtime/"
      },
      {
        "name": "model.track"
      },
      {
        "name": "model.compile"
      },
      {
        "name": "rf.adamw"
      },
      {
        "name": "rf.EarlyStopping",
        "upstream": "https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html"
      },
      {
        "name": "rf.RollbackCheckpoint"
      },
      {
        "name": "rf.EMA"
      }
    ]
  },
  {
    "id": "prediction",
    "title": "Predict, save, and serve",
    "description": "Read Polars prediction results, persist model artifacts, and connect the optional HTTP runtime.",
    "entries": [
      {
        "name": "model.predict",
        "guide": "/concepts/runtime/"
      },
      {
        "name": "model.save",
        "guide": "/advanced/checkpoints-and-state/"
      },
      {
        "name": "rf.Model.load",
        "guide": "/advanced/checkpoints-and-state/"
      },
      {
        "name": "rf.Writer",
        "guide": "/advanced/inference-and-deployment/"
      },
      {
        "name": "rf.postprocess"
      },
      {
        "name": "rf.Postprocessor"
      },
      {
        "name": "rf.Deployment",
        "guide": "/advanced/inference-and-deployment/",
        "extra": "serving"
      },
      {
        "name": "rf.Accelerator",
        "guide": "/advanced/inference-and-deployment/",
        "extra": "serving"
      },
      {
        "name": "rf.JSONBackend",
        "guide": "/advanced/inference-and-deployment/",
        "extra": "serving"
      },
      {
        "name": "rf.Input",
        "extra": "serving"
      },
      {
        "name": "rf.ModelSource",
        "extra": "serving"
      },
      {
        "name": "rf.UpdateOperation",
        "extra": "serving"
      }
    ]
  },
  {
    "id": "mutation",
    "title": "Inspect and evolve a schema",
    "description": "Select nodes and keep the model\u2019s runtime graph synchronized with intentional schema changes.",
    "entries": [
      {
        "name": "rf.Address"
      },
      {
        "name": "rf.where"
      },
      {
        "name": "rf.predicate"
      },
      {
        "name": "model.select"
      },
      {
        "name": "model.update"
      },
      {
        "name": "model.extend"
      },
      {
        "name": "model.delete"
      },
      {
        "name": "model.reset"
      },
      {
        "name": "model.override"
      },
      {
        "name": "rf.MutationLockCallback"
      },
      {
        "name": "rf.RuntimePlacementCallback"
      }
    ]
  },
  {
    "id": "extensions",
    "title": "Extend the modeling pipeline",
    "description": "Implement a field type and use the lower-level encoding, neural computation, and writing stages.",
    "entries": [
      {
        "name": "rf.Extension",
        "guide": "/advanced/custom-data-types/"
      },
      {
        "name": "rf.Component",
        "guide": "/advanced/custom-data-types/"
      },
      {
        "name": "rf.TENSORFIELDS"
      },
      {
        "name": "rf.RequestBase"
      },
      {
        "name": "rf.Leaf"
      },
      {
        "name": "rf.TensorFieldBase"
      },
      {
        "name": "rf.EmbedderBase"
      },
      {
        "name": "rf.DecoderBase"
      },
      {
        "name": "rf.VocabularySyncCallback",
        "guide": "/advanced/online-vocabulary/"
      },
      {
        "name": "model.encode"
      },
      {
        "name": "model.forward"
      },
      {
        "name": "model.write"
      }
    ]
  },
  {
    "id": "contracts",
    "title": "Supporting contracts",
    "description": "Look up the exported types, enums, and protocols used by model, data, training, and extension APIs.",
    "entries": [
      {
        "name": "rf.MaskInput"
      },
      {
        "name": "rf.AttentionMode"
      },
      {
        "name": "rf.AttentionInput"
      },
      {
        "name": "rf.ReductionConfig"
      },
      {
        "name": "rf.Overflow"
      },
      {
        "name": "rf.OverflowInput"
      },
      {
        "name": "rf.SchemaField"
      },
      {
        "name": "rf.TreeFieldInput"
      },
      {
        "name": "rf.NodeAttribute"
      },
      {
        "name": "rf.NodePredicate"
      },
      {
        "name": "rf.NodeSelector"
      },
      {
        "name": "rf.ArrowInput"
      },
      {
        "name": "rf.StratumConfig"
      },
      {
        "name": "rf.Strata"
      },
      {
        "name": "rf.StrataInput"
      },
      {
        "name": "rf.PreprocessorInput"
      },
      {
        "name": "rf.PreprocessorResult"
      },
      {
        "name": "rf.PreprocessorProvider"
      },
      {
        "name": "rf.PostprocessorInput"
      },
      {
        "name": "rf.OptimizerConfig"
      },
      {
        "name": "rf.Scheduler"
      },
      {
        "name": "rf.SchedulerOptions"
      },
      {
        "name": "rf.SchedulerConfig"
      },
      {
        "name": "rf.Metric"
      },
      {
        "name": "rf.Context"
      },
      {
        "name": "rf.BatchContext"
      },
      {
        "name": "rf.TensorInput"
      },
      {
        "name": "rf.RaggedField"
      },
      {
        "name": "rf.TensorKey"
      },
      {
        "name": "rf.Tokens"
      }
    ]
  }
];
