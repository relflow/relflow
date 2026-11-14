/** Authored API index. Sources are pinned; building the docs imports no Python packages. */
interface ApiEntry {
  name: string;
  description: string;
  source: string;
  guide?: string;
  kind?: string;
  extra?: string;
}

interface ApiSection {
  id: string;
  title: string;
  description: string;
  entries: ApiEntry[];
}

const revision = 'a1d0be3e373cf6c183e4926f1bfcbef845f2f883';
const root = `https://github.com/relflow/relflow/tree/${revision}/src/relflow`;

function source(file: string, line: number): string {
  return `https://github.com/relflow/relflow/blob/${revision}/src/relflow/${file}#L${line}`;
}

export const apiSources = {
  "schema": root,
  "fields": root
};

export const apiSections: ApiSection[] = [
  {
    "id": "schema",
    "title": "Build a model",
    "description": "Declare records, choose architecture defaults, and configure how context reaches prediction targets.",
    "entries": [
      {
        "name": "rf.Model",
        "description": "Construct an untrained neural model from a reusable class schema, named fields, or a resolved schema.",
        "source": source('architecture/root.py', 113),
        "guide": "/concepts/models/",
        "kind": "class"
      },
      {
        "name": "rf.Branch",
        "description": "Declare a collection of related records, with its own context settings and optional retained-length limit.",
        "source": source('structs/structure.py', 49),
        "guide": "/concepts/branches/",
        "kind": "class"
      },
      {
        "name": "rf.Schema",
        "description": "Inspect the resolved field tree, requests, objectives, and output roles used to construct a model.",
        "source": source('structs/experiment.py', 108),
        "kind": "class"
      },
      {
        "name": "rf.Model.from_yaml",
        "description": "Construct a fresh model from a YAML schema; this does not restore trained weights.",
        "source": source('architecture/root.py', 340),
        "kind": "classmethod"
      },
      {
        "name": "rf.Mask",
        "description": "Select coordinates to hide, drop, or reconstruct, including query-selected and sampled targets.",
        "source": source('structs/tree.py', 19),
        "guide": "/concepts/targets/",
        "kind": "configuration"
      },
      {
        "name": "rf.Jitter",
        "description": "Configure training-only affine noise for supported continuous field representations.",
        "source": source('tensorfields/shared/jitter.py', 9),
        "kind": "configuration"
      },
      {
        "name": "rf.Attention",
        "description": "Reduce a branch to a configured number of learned context vectors.",
        "source": source('structs/reduction.py', 26),
        "kind": "configuration"
      },
      {
        "name": "rf.Mean",
        "description": "Reduce a branch by averaging its present encoded coordinates.",
        "source": source('structs/reduction.py', 13),
        "kind": "configuration"
      },
      {
        "name": "rf.presets",
        "description": "Access named architecture policies, optional fit defaults, and the custom-policy registry.",
        "source": source('presets/__init__.py', 1),
        "guide": "/concepts/runtime/",
        "kind": "module"
      },
      {
        "name": "rf.presets.SMALL",
        "description": "Use the bundled small policy as a registered architecture or fit recipe.",
        "source": source('presets/builtins.py', 15),
        "guide": "/concepts/runtime/",
        "kind": "preset"
      },
      {
        "name": "rf.presets.MEDIUM",
        "description": "Use the bundled medium policy as a registered architecture or fit recipe.",
        "source": source('presets/builtins.py', 32),
        "guide": "/concepts/runtime/",
        "kind": "preset"
      },
      {
        "name": "rf.presets.get",
        "description": "Look up a registered policy by name.",
        "source": source('presets/registry.py', 11),
        "kind": "function"
      },
      {
        "name": "rf.presets.register",
        "description": "Validate and register a custom policy under a unique name.",
        "source": source('presets/registry.py', 21),
        "kind": "function"
      },
      {
        "name": "rf.presets.Preset",
        "description": "Define a named policy with model width, node defaults, and optional training defaults.",
        "source": source('presets/base.py', 60),
        "kind": "configuration"
      },
      {
        "name": "rf.presets.BranchDefaults",
        "description": "Set architecture defaults for the root and child branches of a policy.",
        "source": source('presets/base.py', 19),
        "kind": "configuration"
      },
      {
        "name": "rf.presets.LeafDefaults",
        "description": "Set the shared decoder defaults used by a policy\u2019s field requests.",
        "source": source('presets/base.py', 32),
        "kind": "configuration"
      },
      {
        "name": "rf.presets.FitDefaults",
        "description": "Define training settings applied only when a fit explicitly selects the policy.",
        "source": source('presets/base.py', 45),
        "kind": "configuration"
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
        "description": "Model numeric magnitude and predict a scalar in its original units.",
        "source": source('tensorfields/extensions/number.py', 163),
        "guide": "/data-types/number/",
        "kind": "field type"
      },
      {
        "name": "rf.Quantile",
        "description": "Represent a numeric value by its position in a learned training distribution.",
        "source": source('tensorfields/extensions/quantile.py', 138),
        "guide": "/data-types/quantile/",
        "kind": "field type"
      },
      {
        "name": "rf.Ordinal",
        "description": "Represent ordered labels or integer ranks and predict a discrete rank or label.",
        "source": source('tensorfields/extensions/ordinal.py', 35),
        "guide": "/data-types/ordinal/",
        "kind": "field type"
      },
      {
        "name": "rf.Boolean",
        "description": "Represent true-or-false facts and predict the probability of True.",
        "source": source('tensorfields/extensions/boolean.py', 50),
        "guide": "/data-types/boolean/",
        "kind": "field type"
      },
      {
        "name": "rf.Category",
        "description": "Learn a vocabulary of scalar labels for input context or classification.",
        "source": source('tensorfields/extensions/category.py', 59),
        "guide": "/data-types/category/",
        "kind": "field type"
      },
      {
        "name": "rf.Enum",
        "description": "Declare the fixed set of allowed labels for a scalar field.",
        "source": source('tensorfields/extensions/enum.py', 53),
        "guide": "/data-types/enum/",
        "kind": "field type"
      },
      {
        "name": "rf.Set",
        "description": "Represent unordered label membership and predict independent membership probabilities.",
        "source": source('tensorfields/extensions/set.py', 56),
        "guide": "/data-types/set/",
        "kind": "field type"
      },
      {
        "name": "rf.Hash",
        "description": "Preserve identifier matching relationships without a learned identity vocabulary.",
        "source": source('tensorfields/extensions/hashable.py', 64),
        "guide": "/data-types/hash/",
        "kind": "field type"
      },
      {
        "name": "rf.Group",
        "description": "Learn shared group representations for repeated identities through downstream prediction tasks.",
        "source": source('tensorfields/extensions/group.py', 111),
        "guide": "/data-types/group/",
        "kind": "field type"
      },
      {
        "name": "rf.Period",
        "description": "Encode recurring calendar context such as weekday, month, or hour.",
        "source": source('tensorfields/extensions/dateparts.py', 251),
        "guide": "/data-types/period/",
        "kind": "field type"
      },
      {
        "name": "rf.Duration",
        "description": "Derive signed elapsed time from timestamp endpoints and model it in fixed units.",
        "source": source('tensorfields/extensions/duration.py', 23),
        "guide": "/data-types/duration/",
        "kind": "field type"
      },
      {
        "name": "rf.Text",
        "description": "Use a frozen text encoder\u2019s representations as context for a predictive model.",
        "source": source('tensorfields/extensions/text.py', 166),
        "guide": "/data-types/text/",
        "kind": "field type"
      },
      {
        "name": "rf.Vector",
        "description": "Represent fixed-width numeric coordinates as inputs or prediction targets.",
        "source": source('tensorfields/extensions/vector.py', 54),
        "guide": "/data-types/vector/",
        "kind": "field type"
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
        "description": "Load named splits from Arrow tables, batches, datasets, files, or restartable Arrow factories.",
        "source": source('data/datasets/arrow.py', 691),
        "guide": "/advanced/data-modules/",
        "kind": "class"
      },
      {
        "name": "rf.PolarsDataModule",
        "description": "Convert eager Polars DataFrames once, then use the shared Arrow data pipeline.",
        "source": source('data/datasets/polars.py', 15),
        "kind": "class"
      },
      {
        "name": "rf.DuckDBDataModule",
        "description": "Stream configured SQL queries as restartable Arrow data splits.",
        "source": source('data/datasets/duckdb.py', 64),
        "kind": "class"
      },
      {
        "name": "rf.CustomDataModule",
        "description": "Adapt restartable mapping-based IterableDataset splits to bounded Arrow batches.",
        "source": source('data/datasets/custom.py', 115),
        "guide": "/advanced/custom-data-modules/",
        "kind": "class"
      },
      {
        "name": "rf.SyntheticDataModule",
        "description": "Adapt zero-argument record factories to restartable Arrow batches.",
        "source": source('data/datasets/synthetic.py', 21),
        "kind": "class"
      },
      {
        "name": "rf.source",
        "description": "Describe file paths, directories, or globs with format, filesystem, and partition-reading options.",
        "source": source('data/sources.py', 290),
        "kind": "function"
      },
      {
        "name": "rf.preprocess",
        "description": "Decorate an eager Polars transform that prepares source observations before structural queries and encoding.",
        "source": source('data/processors.py', 398),
        "guide": "/advanced/data-modules/",
        "kind": "decorator"
      },
      {
        "name": "rf.Preprocessor",
        "description": "Configure a decorated input transform, including its scope and bound application parameters.",
        "source": source('data/processors.py', 281),
        "kind": "class"
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
        "description": "Describe a YAML schema or checkpoint on CPU without building its neural graph.",
        "source": source('inspection.py', 100),
        "guide": "/advanced/command-line/",
        "kind": "function"
      },
      {
        "name": "rf.check",
        "description": "Check declarations and data sources; execute=True opts into sampled preprocessing and tensorization.",
        "source": source('inspection.py', 107),
        "guide": "/advanced/command-line/",
        "kind": "function"
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
        "description": "Train the current model; select fit defaults or optimizer settings and use ckpt_path for full training-state resume.",
        "source": source('architecture/root.py', 686),
        "guide": "/concepts/runtime/",
        "kind": "method"
      },
      {
        "name": "model.validate",
        "description": "Evaluate the validation split and return aggregate metric dictionaries.",
        "source": source('architecture/root.py', 840),
        "guide": "/concepts/runtime/",
        "kind": "method"
      },
      {
        "name": "model.test",
        "description": "Evaluate the test split and return aggregate metric dictionaries.",
        "source": source('architecture/root.py', 867),
        "guide": "/concepts/runtime/",
        "kind": "method"
      },
      {
        "name": "model.track",
        "description": "Log a custom metric through the model\u2019s Lightning logging path.",
        "source": source('architecture/root.py', 582),
        "kind": "method"
      },
      {
        "name": "model.compile",
        "description": "Selectively and lazily compile compatible encoders and attention pools in place.",
        "source": source('architecture/root.py', 408),
        "kind": "method"
      },
      {
        "name": "rf.adamw",
        "description": "Create an AdamW factory for custom optimizer configuration or direct Lightning workflows.",
        "source": source('helpers/optimizers.py', 32),
        "kind": "function"
      },
      {
        "name": "rf.EarlyStopping",
        "description": "Configure Lightning early stopping for a monitored training or validation metric.",
        "source": source('__init__.py', 11),
        "kind": "callback"
      },
      {
        "name": "rf.RollbackCheckpoint",
        "description": "Track the best fit checkpoint and restore its model state at fit completion when enabled.",
        "source": source('architecture/checkpoint.py', 23),
        "kind": "callback"
      },
      {
        "name": "rf.EMA",
        "description": "Evaluate, select checkpoints, and finish fitting with averaged learned parameters when enabled.",
        "source": source('architecture/averaging.py', 27),
        "kind": "callback"
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
        "description": "Return an eager Polars DataFrame of named prediction results, or stream through callbacks with return_predictions=False.",
        "source": source('architecture/root.py', 1005),
        "guide": "/concepts/runtime/",
        "kind": "method"
      },
      {
        "name": "model.save",
        "description": "Save the resolved schema and learned model state; full optimizer resume uses a training checkpoint.",
        "source": source('architecture/root.py', 632),
        "guide": "/advanced/checkpoints-and-state/",
        "kind": "method"
      },
      {
        "name": "rf.Model.load",
        "description": "Restore a saved model artifact, including its field vocabularies and numeric state.",
        "source": source('architecture/root.py', 903),
        "guide": "/advanced/checkpoints-and-state/",
        "kind": "classmethod"
      },
      {
        "name": "rf.Writer",
        "description": "Stream batch prediction output to Parquet or Arrow IPC files without collecting the full result in memory.",
        "source": source('inference/callback.py', 66),
        "guide": "/advanced/inference-and-deployment/",
        "kind": "callback"
      },
      {
        "name": "rf.postprocess",
        "description": "Decorate an eager Polars transform applied after prediction writing.",
        "source": source('data/processors.py', 431),
        "kind": "decorator"
      },
      {
        "name": "rf.Postprocessor",
        "description": "Configure a decorated transform of decoded prediction frames.",
        "source": source('data/processors.py', 336),
        "kind": "class"
      },
      {
        "name": "rf.Deployment",
        "description": "Configure model loading, request batching, and an HTTP prediction application.",
        "source": source('inference/deployment.py', 400),
        "guide": "/advanced/inference-and-deployment/",
        "kind": "class",
        "extra": "serving"
      },
      {
        "name": "rf.Accelerator",
        "description": "Choose automatic device selection, CPU, CUDA, or MPS for serving.",
        "source": source('inference/deployment.py', 50),
        "guide": "/advanced/inference-and-deployment/",
        "kind": "enum",
        "extra": "serving"
      },
      {
        "name": "rf.JSONBackend",
        "description": "Choose the standard or optional fast JSON response encoder.",
        "source": source('inference/deployment.py', 69),
        "guide": "/advanced/inference-and-deployment/",
        "kind": "enum",
        "extra": "serving"
      },
      {
        "name": "rf.Input",
        "description": "Type a single HTTP input record as a mapping of field names to values.",
        "source": source('inference/deployment.py', 38),
        "kind": "type alias",
        "extra": "serving"
      },
      {
        "name": "rf.ModelSource",
        "description": "Type a serving model source as a checkpoint path or an in-memory model.",
        "source": source('inference/deployment.py', 39),
        "kind": "type alias",
        "extra": "serving"
      },
      {
        "name": "rf.UpdateOperation",
        "description": "Type one queued schema update for deployment startup.",
        "source": source('inference/deployment.py', 40),
        "kind": "type alias",
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
        "name": "rf.Address",
        "description": "Create an absolute schema address from child-name segments or a serialized path.",
        "source": source('structs/tree.py', 130),
        "kind": "class"
      },
      {
        "name": "rf.where",
        "description": "Start a composable predicate against a schema-node attribute.",
        "source": source('structs/selectors.py', 146),
        "kind": "function"
      },
      {
        "name": "rf.predicate",
        "description": "Wrap an application callable for composition with schema predicates.",
        "source": source('structs/selectors.py', 52),
        "kind": "function"
      },
      {
        "name": "model.select",
        "description": "Return schema nodes matching the supplied selectors.",
        "source": source('architecture/root.py', 452),
        "kind": "method"
      },
      {
        "name": "model.update",
        "description": "Change declared node options and rebuild runtime modules while retaining compatible state.",
        "source": source('architecture/root.py', 460),
        "kind": "method"
      },
      {
        "name": "model.extend",
        "description": "Append named children to one selected branch and rebuild the model graph.",
        "source": source('architecture/root.py', 485),
        "kind": "method"
      },
      {
        "name": "model.delete",
        "description": "Remove selected nodes and rebuild the model graph.",
        "source": source('architecture/root.py', 495),
        "kind": "method"
      },
      {
        "name": "model.reset",
        "description": "Reinitialize selected runtime modules while keeping their schema declarations.",
        "source": source('architecture/root.py', 503),
        "kind": "method"
      },
      {
        "name": "model.override",
        "description": "Temporarily apply schema changes within a context manager.",
        "source": source('architecture/root.py', 516),
        "kind": "method"
      },
      {
        "name": "rf.MutationLockCallback",
        "description": "Prevent runtime schema mutations while Lightning owns an active model loop.",
        "source": source('architecture/mutations.py', 49),
        "kind": "callback"
      },
      {
        "name": "rf.RuntimePlacementCallback",
        "description": "Place modules created after initialization on the model\u2019s active device.",
        "source": source('architecture/mutations.py', 97),
        "kind": "callback"
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
        "description": "Register a field type\u2019s accepted values and role-specific encoding, learning, and output components.",
        "source": source('tensorfields/base.py', 785),
        "guide": "/advanced/custom-data-types/",
        "kind": "registry"
      },
      {
        "name": "rf.Component",
        "description": "Select an explicit extension role such as Request, Embedder, derive, loss, or write.",
        "source": source('structs/enums.py', 99),
        "guide": "/advanced/custom-data-types/",
        "kind": "enum"
      },
      {
        "name": "rf.TENSORFIELDS",
        "description": "Inspect the live registry of registered tensorfield extensions.",
        "source": source('tensorfields/base.py', 782),
        "kind": "registry"
      },
      {
        "name": "rf.RequestBase",
        "description": "Define validated configuration fields for a custom tensorfield request.",
        "source": source('tensorfields/base.py', 56),
        "kind": "extension base"
      },
      {
        "name": "rf.Leaf",
        "description": "Access the common schema-node base underlying field requests.",
        "source": source('structs/tree.py', 255),
        "kind": "extension base"
      },
      {
        "name": "rf.TensorFieldBase",
        "description": "Implement tensorized field values, targets, and field-owned content.",
        "source": source('tensorfields/base.py', 703),
        "kind": "extension base"
      },
      {
        "name": "rf.EmbedderBase",
        "description": "Implement the neural representation passed from a field into branch context.",
        "source": source('tensorfields/base.py', 213),
        "kind": "extension base"
      },
      {
        "name": "rf.DecoderBase",
        "description": "Implement field-coordinate decoding with extension-owned output heads.",
        "source": source('tensorfields/base.py', 286),
        "kind": "extension base"
      },
      {
        "name": "rf.VocabularySyncCallback",
        "description": "Synchronize distributed vocabulary state and report low exposure before evaluation.",
        "source": source('tensorfields/shared/vocabulary.py', 614),
        "guide": "/advanced/online-vocabulary/",
        "kind": "callback"
      },
      {
        "name": "model.encode",
        "description": "Preprocess and tensorize Arrow inputs for an explicit data stratum.",
        "source": source('architecture/root.py', 921),
        "kind": "method"
      },
      {
        "name": "model.forward",
        "description": "Run neural computation on encoded model inputs and return field prediction objects.",
        "source": source('architecture/root.py', 639),
        "kind": "method"
      },
      {
        "name": "model.write",
        "description": "Materialize field predictions and source geometry as a Polars DataFrame with the configured public outputs.",
        "source": source('architecture/root.py', 908),
        "kind": "method"
      }
    ]
  },
  {
    "id": "contracts",
    "title": "Supporting contracts",
    "description": "Look up the exported types, enums, and protocols used by model, data, training, and extension APIs.",
    "entries": [
      {
        "name": "rf.MaskInput",
        "description": "Accepted visibility and reconstruction policy forms.",
        "source": source('structs/tree.py', 103),
        "kind": "type alias"
      },
      {
        "name": "rf.AttentionMode",
        "description": "Available branch attention modes.",
        "source": source('structs/enums.py', 72),
        "kind": "enum"
      },
      {
        "name": "rf.AttentionInput",
        "description": "Accepted attention enum, string, or disabled configuration.",
        "source": source('structs/enums.py', 94),
        "kind": "type alias"
      },
      {
        "name": "rf.ReductionConfig",
        "description": "Accepted attention or mean branch-reduction configuration.",
        "source": source('structs/reduction.py', 47),
        "kind": "type alias"
      },
      {
        "name": "rf.Overflow",
        "description": "Available policies for collections exceeding a finite branch limit.",
        "source": source('structs/enums.py', 87),
        "kind": "enum"
      },
      {
        "name": "rf.OverflowInput",
        "description": "Accepted overflow enum or string forms.",
        "source": source('structs/enums.py', 95),
        "kind": "type alias"
      },
      {
        "name": "rf.SchemaField",
        "description": "A bound branch or field-request schema node.",
        "source": source('structs/selectors.py', 14),
        "kind": "type alias"
      },
      {
        "name": "rf.TreeFieldInput",
        "description": "A branch or field declaration accepted when constructing a schema.",
        "source": source('structs/experiment.py', 40),
        "kind": "type alias"
      },
      {
        "name": "rf.NodeAttribute",
        "description": "A queryable node attribute produced by where.",
        "source": source('structs/selectors.py', 69),
        "kind": "class"
      },
      {
        "name": "rf.NodePredicate",
        "description": "A composable predicate for selecting schema nodes.",
        "source": source('structs/selectors.py', 17),
        "kind": "class"
      },
      {
        "name": "rf.NodeSelector",
        "description": "Accepted predicate, attribute, or callable node selector.",
        "source": source('structs/selectors.py', 151),
        "kind": "type alias"
      },
      {
        "name": "rf.ArrowInput",
        "description": "Accepted Arrow objects, restartable sources, datasets, or file paths.",
        "source": source('data/datasets/arrow.py', 41),
        "kind": "type alias"
      },
      {
        "name": "rf.StratumConfig",
        "description": "One shared option value or a per-stratum option mapping.",
        "source": source('data/datasets/base.py', 15),
        "kind": "type alias"
      },
      {
        "name": "rf.Strata",
        "description": "The train, validate, test, and predict phases.",
        "source": source('structs/enums.py', 19),
        "kind": "enum"
      },
      {
        "name": "rf.StrataInput",
        "description": "Accepted data-phase enum or string forms.",
        "source": source('structs/enums.py', 96),
        "kind": "type alias"
      },
      {
        "name": "rf.PreprocessorInput",
        "description": "Accepted input-preprocessor instance or ordered collection.",
        "source": source('data/processors.py', 350),
        "kind": "type alias"
      },
      {
        "name": "rf.PreprocessorResult",
        "description": "An eager Polars frame, iterable of frames, or no emitted frame.",
        "source": source('data/processors.py', 22),
        "kind": "type alias"
      },
      {
        "name": "rf.PreprocessorProvider",
        "description": "Pipeline-owned parameter names available to input transforms.",
        "source": source('data/processors.py', 26),
        "kind": "enum"
      },
      {
        "name": "rf.PostprocessorInput",
        "description": "Accepted prediction-postprocessor instance or ordered collection.",
        "source": source('data/processors.py', 351),
        "kind": "type alias"
      },
      {
        "name": "rf.OptimizerConfig",
        "description": "An optimizer instance or model-bound optimizer factory.",
        "source": source('architecture/root.py', 65),
        "kind": "type alias"
      },
      {
        "name": "rf.Scheduler",
        "description": "Stateful optimizer scheduler protocol.",
        "source": source('architecture/root.py', 68),
        "kind": "protocol"
      },
      {
        "name": "rf.SchedulerOptions",
        "description": "Lightning scheduler controls such as interval, frequency, and monitor.",
        "source": source('architecture/root.py', 82),
        "kind": "typed mapping"
      },
      {
        "name": "rf.SchedulerConfig",
        "description": "Accepted scheduler instance, options, or optimizer-bound factory.",
        "source": source('architecture/root.py', 94),
        "kind": "type alias"
      },
      {
        "name": "rf.Metric",
        "description": "Canonical metric names used by built-in learning objectives.",
        "source": source('structs/enums.py', 59),
        "kind": "enum"
      },
      {
        "name": "rf.Context",
        "description": "Field-owned encoding resources and batch hash salt.",
        "source": source('tensorfields/base.py', 198),
        "kind": "data contract"
      },
      {
        "name": "rf.BatchContext",
        "description": "Protocol for isolating worker batch resources and carrying field bindings.",
        "source": source('tensorfields/base.py', 95),
        "kind": "protocol"
      },
      {
        "name": "rf.Derive",
        "description": "Protocol for deriving one Arrow value per retained owner record before observation and masking.",
        "source": source('tensorfields/base.py', 76),
        "kind": "protocol"
      },
      {
        "name": "rf.TensorInput",
        "description": "Compact model input presented to a tensorfield embedder.",
        "source": source('tensorfields/base.py', 206),
        "kind": "data contract"
      },
      {
        "name": "rf.RaggedField",
        "description": "Arrow-backed retained values with dense model geometry and structural alignment.",
        "source": source('data/ragged.py', 140),
        "kind": "data contract"
      },
      {
        "name": "rf.TensorKey",
        "description": "Canonical keys for field state, content, targets, and prediction payloads.",
        "source": source('structs/enums.py', 44),
        "kind": "enum"
      },
      {
        "name": "rf.Tokens",
        "description": "Canonical value states such as valued, null, padded, masked, and other.",
        "source": source('structs/enums.py', 11),
        "kind": "enum"
      }
    ]
  }
];
