<script setup lang="ts">
import CrossValidationReport from "@/components/CrossValidationReport.vue";
import DataFrameWidget from "@/components/DataFrameWidget.vue";
import HtmlSnippetWidget from "@/components/HtmlSnippetWidget.vue";
import ImageWidget from "@/components/ImageWidget.vue";
import MarkdownWidget from "@/components/MarkdownWidget.vue";
import PlotlyWidget from "@/components/PlotlyWidget.vue";
import VegaWidget from "@/components/VegaWidget.vue";
import type { PresentableItem } from "@/models";

const props = defineProps<{ item: PresentableItem }>();

function matchMediaType(mediaType: string) {
  return props.item.mediaType.startsWith(mediaType);
}
</script>

<template>
  <DataFrameWidget
    v-if="matchMediaType('application/vnd.dataframe')"
    :columns="props.item.data.columns"
    :data="props.item.data.data"
    :index="props.item.data.index"
    :index-names="props.item.data.index_names"
  />
  <ImageWidget
    v-if="matchMediaType('image/')"
    :mediaType="props.item.mediaType"
    :base64-src="props.item.data"
    :alt="props.item.name"
  />
  <MarkdownWidget
    v-if="matchMediaType('text/markdown') || matchMediaType('application/json')"
    :source="props.item.data"
  />
  <VegaWidget v-if="matchMediaType('application/vnd.vega.v5+json')" :spec="props.item.data" />
  <PlotlyWidget v-if="matchMediaType('application/vnd.plotly.v1+json')" :spec="props.item.data" />
  <HtmlSnippetWidget
    v-if="matchMediaType('application/vnd.sklearn.estimator+html')"
    :src="props.item.data"
  />
  <HtmlSnippetWidget v-if="matchMediaType('text/html')" :src="props.item.data" />
  <CrossValidationReport
    v-if="matchMediaType('application/vnd.skore.cross_validation')"
    :scalar-results="props.item.data.scalar_results"
    :tabular-results="props.item.data.tabular_results"
    :plots="props.item.data.plots"
    :sections="props.item.data.sections"
  />
</template>
