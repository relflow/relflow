import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import { hostedURLs } from './src/site/build.mjs';

const address = new URL(process.env.RELFLOW_DOCS_URL || 'https://www.relflow.ai/');
const revision = process.env.RELFLOW_DOCS_REVISION;
if (revision && !/^[a-f0-9]{40}$/.test(revision))
  throw new Error('RELFLOW_DOCS_REVISION must be the complete checkout Git revision.');

export default defineConfig({
  site: address.origin,
  base: address.pathname,
  output: 'static',
  trailingSlash: 'always',
  devToolbar: { enabled: false },
  integrations: [
    starlight({
      title: 'relflow',
      tableOfContents: false,
      description:
        'Build predictive neural models from typed Python schemas, nested records, and histories.',
      favicon: '/favicon.svg',
      logo: {
        light: './src/assets/branding/header.light.svg',
        dark: './src/assets/branding/header.dark.svg',
        replacesTitle: true,
        alt: '',
      },
      components: {
        Sidebar: './src/components/Sidebar.astro',
        EditLink: './src/components/SourceLink.astro',
      },
      ...(revision
        ? {
            editLink: {
              baseUrl: `https://github.com/relflow/relflow/blob/${revision}/docs/starlight/`,
            },
          }
        : {}),
      social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/relflow/relflow' }],
      customCss: ['./src/styles/site.css', './src/code/notes.css'],
      sidebar: [
        {
          label: 'Get started',
          items: [
            { label: 'What is relflow?', slug: '' },
            { label: 'Featureless modeling', slug: 'featureless' },
            { label: 'Modular modeling', slug: 'modular' },
            { label: 'Installation', slug: 'installation' },
          ],
        },
        {
          label: 'Use your data',
          items: [
            { label: 'Define one example', slug: 'data' },
            { label: 'Example datasets', slug: 'datasets' },
            { label: 'Choose your types', slug: 'data-types' },
            { label: 'Choose your predictions', slug: 'predictions' },
          ],
        },
        {
          label: 'Core Concepts',
          items: [
            { label: 'Models & schemas', slug: 'concepts/models' },
            { label: 'Data types', slug: 'concepts/data-types' },
            { label: 'Branches & context', slug: 'concepts/branches' },
            { label: 'Targets & masking', slug: 'concepts/targets' },
            { label: 'Data modules', slug: 'concepts/data-modules' },
            { label: 'Training & learned state', slug: 'concepts/training' },
            { label: 'Regularization', slug: 'concepts/regularization' },
            { label: 'Predictions', slug: 'concepts/predictions' },
          ],
        },
        {
          label: 'Data types',
          items: [
            { label: 'Number', slug: 'data-types/number' },
            { label: 'Quantile', slug: 'data-types/quantile' },
            { label: 'Ordinal', slug: 'data-types/ordinal' },
            { label: 'Boolean', slug: 'data-types/boolean' },
            { label: 'Category', slug: 'data-types/category' },
            { label: 'Enum', slug: 'data-types/enum' },
            { label: 'Set', slug: 'data-types/set' },
            { label: 'Hash', slug: 'data-types/hash' },
            { label: 'Group', slug: 'data-types/group' },
            { label: 'Period', slug: 'data-types/period' },
            { label: 'Duration', slug: 'data-types/duration' },
            { label: 'Text', slug: 'data-types/text' },
            { label: 'Vector', slug: 'data-types/vector' },
          ],
        },
        {
          label: 'Advanced Usage',
          items: [
            { label: 'Overview', slug: 'advanced' },
            { label: 'Custom data modules', slug: 'advanced/custom-data-modules' },
            { label: 'Custom data types', slug: 'advanced/custom-data-types' },
            { label: 'Online vocabulary', slug: 'advanced/online-vocabulary' },
            { label: 'Mutations', slug: 'advanced/mutations' },
            { label: 'Embeddings', slug: 'advanced/embeddings' },
            { label: 'Distributed training & streaming', slug: 'advanced/distributed-data' },
            { label: 'Checkpoints & model state', slug: 'advanced/checkpoints-and-state' },
            { label: 'Command line', slug: 'advanced/command-line' },
            { label: 'Masking & pretraining', slug: 'advanced/masking-and-pretraining' },
            { label: 'Inference & deployment', slug: 'advanced/inference-and-deployment' },
          ],
        },
        { label: 'API map', slug: 'api-map' },
        { label: 'Benchmarks', slug: 'benchmarks' },
        { label: 'Evidence', slug: 'evidence' },
      ],
      pagination: true,
      disable404Route: true,
      credits: false,
    }),
    hostedURLs(address.pathname),
  ],
});
