<script setup lang="ts">
import "datatables.net-dt/css/dataTables.dataTables.min.css";

import DataTablesCore from "datatables.net";
import DataTable from "datatables.net-vue3";

DataTable.use(DataTablesCore);

const props = defineProps<{
  headers: string[];
  values: any[][];
}>();
</script>

<template>
  <DataTable :data="props.values">
    <thead>
      <tr>
        <th v-for="(name, index) in props.headers" :key="index">{{ name }}</th>
      </tr>
    </thead>
  </DataTable>
</template>
