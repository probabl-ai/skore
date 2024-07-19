<script setup lang="ts">
import { type DataStore } from "../models";

const props = defineProps<{ dataStore: DataStore }>();
</script>

<template>
  <h1>{{ props.dataStore.path }}</h1>
  <aside>
    <h2>Views</h2>
    <div v-for="(value, name, index) in props.dataStore.views" :key="index">
      <h3>{{ name }}</h3>
      <div v-html="value" />
    </div>
  </aside>
  <aside>
    <h2>Logs</h2>
    <div v-for="(value, name, index) in props.dataStore.logs" :key="index">
      <h3>{{ name }}</h3>
      <div v-html="value" />
    </div>
  </aside>
  <aside>
    <h2>Artifacts</h2>
    <div v-for="(value, name, index) in props.dataStore.artifacts" :key="index">
      <h3>{{ name }}</h3>
      <div v-html="value.path" />
    </div>
  </aside>
  <aside>
    <h2>Info</h2>
    <div v-for="(value, name, index) in props.dataStore.info" :key="index">
      <h3>{{ name }}</h3>
      <div v-html="value" />
    </div>
  </aside>
</template>

<style scoped>
aside {
  border: 1px solid #e9e9e9;
  margin: 5px;

  h2 {
    background-color: #f2f1f1;
    color: #7e7e7e;
    font-weight: 300;
  }

  & > div {
    padding: 5px;
  }
}
</style>
