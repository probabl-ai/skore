<script setup lang="ts">
import { inject } from "vue";

const props = defineProps<{ name: string }>();
const selectedTabName = inject("selectedTabName");
</script>

<template>
  <div class="tab-contend" v-if="props.name === selectedTabName">
    <slot />
  </div>
</template>
