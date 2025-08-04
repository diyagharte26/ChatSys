from kubernetes import client, config
from kubernetes.client.rest import ApiException
from typing import Optional
from langchain.agents import Tool
import os
import yaml 
import base64 

# Load Kubernetes configuration.
# This will load the configuration from the default location (~/.kube/config.merged)
# or from the in-cluster service account environment.

kube_config_path = os.path.expanduser("~/.kube/config.merged")
try:
    config.load_kube_config(config_file=kube_config_path)
except config.ConfigException:
    try:
        config.load_incluster_config()
    except config.ConfigException as e:
        raise Exception("Could not configure kubernetes client") from e


v1 = client.CoreV1Api()
apps_v1 = client.AppsV1Api()
autoscaling_v1 = client.AutoscalingV1Api() # For HPA resources
networking_v1 = client.NetworkingV1Api() # For Ingress resources

def create_pod(name: str, image: str, namespace: str = "default", port: Optional[int] = None) -> dict:
    """Create a Pod with name, image, and optional port."""
    try:
        pod_manifest = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": name},
            "spec": {
                "containers": [{"name": name, "image": image}]
            }
        }
        if port:
            pod_manifest["spec"]["containers"][0]["ports"] = [{"containerPort": port}]
        
        v1.create_namespaced_pod(body=pod_manifest, namespace=namespace)
        
        return {
            "status": "success",
            "resource": "Pod",
            "name": name,
            "namespace": namespace
        }

    except ApiException as e:
        return {
            "status": "error",
            "resource": "Pod",
            "name": name,
            "namespace": namespace,
            "reason": str(e)
        }

def del_pod(name: str, namespace: str = "default", confirm: bool = False) -> str:
    """Delete a Kubernetes Pod by name and namespace. Requires confirmation."""
    try:
        if not confirm:
            return f"Please confirm deletion of pod '{name}' in namespace '{namespace}' by setting 'confirm=true'."
        
        v1.delete_namespaced_pod(name=name, namespace=namespace)
        return f"Pod '{name}' deleted from namespace '{namespace}'."
    
    except ApiException as e:
        if e.status == 404:
            return f"Pod '{name}' not found in namespace '{namespace}'."
        return f"Error deleting pod '{name}': {e}"


def create_ns(name: str) -> str:
    """Create a Kubernetes Namespace with the given name."""
    try:
        namespace_manifest = {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {"name": name}
        }
        v1.create_namespace(body=namespace_manifest)
        return f"Namespace '{name}' created."
    except ApiException as e:
        return f"Error creating namespace '{name}': {e}"


def del_ns(name: str, confirm: bool = False) -> str:
    """Delete a Kubernetes Namespace by name. Requires confirmation."""
    try:
        if not confirm:
            return f"Please confirm deletion of namespace '{name}' by setting 'confirm=true'."
        v1.delete_namespace(name=name)
        return f"Namespace '{name}' deleted."
    except ApiException as e:
        if e.status == 404:
            return f"Namespace '{name}' not found."
        return f"Error deleting namespace '{name}': {e}"


def create_deploy(name: str, image: str, replicas: int = 1, ns: str = "default", port: Optional[int] = None) -> str:
    """Create a Kubernetes Deployment with name, image, replica count, and optional port."""
    try:
        manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": name},
            "spec": {
                "replicas": replicas,
                "selector": {"matchLabels": {"app": name}},
                "template": {
                    "metadata": {"labels": {"app": name}},
                    "spec": {
                        "containers": [
                            {"name": name, "image": image}
                        ]
                    }
                }
            }
        }
        if port:
            manifest["spec"]["template"]["spec"]["containers"][0]["ports"] = [{"containerPort": port}]
        apps_v1.create_namespaced_deployment(body=manifest, namespace=ns)
        return f"Deployment '{name}' created in namespace '{ns}'."
    except ApiException as e:
        return f"Error creating deployment '{name}': {e}"


def delete_deploy(name: str, ns: str = "default", confirm: bool = False) -> str:
    """Delete a Kubernetes Deployment with confirmation."""
    try:
        if not confirm:
            return f"Confirm deletion of deployment '{name}' in namespace '{ns}' by setting confirm=True."
        apps_v1.delete_namespaced_deployment(name=name, namespace=ns)
        return f"Deployment '{name}' deleted from namespace '{ns}'."
    except ApiException as e:
        if e.status == 404:
            return f"Deployment '{name}' not found in namespace '{ns}'."
        return f"Error deleting deployment '{name}': {e}"

def create_svc(name: str, app: str, port: int, tgt: int, ns: str = "default", typ: str = "ClusterIP") -> str:
    """Create a Kubernetes Service with name, selector app, port, target port, namespace, and type."""
    try:
        manifest = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": name},
            "spec": {
                "selector": {"app": app},
                "ports": [{"protocol": "TCP", "port": port, "targetPort": tgt}],
                "type": typ
            }
        }
        v1.create_namespaced_service(body=manifest, namespace=ns)
        return f"Service '{name}' created in namespace '{ns}'."
    except ApiException as e:
        return f"Error creating service '{name}': {e}"

def delete_svc(name: str, ns: str = "default", cfm: bool = False) -> str:
    """Delete a Kubernetes Service by name and namespace."""
    try:
        if not cfm:
            return f"Confirm deletion of service '{name}' in ns '{ns}' by setting cfm=True."
        v1.delete_namespaced_service(name=name, namespace=ns)
        return f"Service '{name}' deleted from ns '{ns}'."
    except ApiException as e:
        if e.status == 404:
            return f"Service '{name}' not found in ns '{ns}'."
        return f"Error deleting service '{name}': {e}"

def list_svc(ns: Optional[str] = None) -> dict:
    """List Kubernetes services in a namespace or all namespaces."""
    try:
        if ns:
            svcs = [s.metadata.name for s in v1.list_namespaced_service(namespace=ns).items]
            return {
                "status": "success",
                "scope": ns,
                "services": svcs
            }
        else:
            svcs = [s.metadata.name for s in v1.list_service_for_all_namespaces().items]
            return {
                "status": "success",
                "scope": "all",
                "services": svcs
            }
    except ApiException as e:
        return {
            "status": "error",
            "message": f"Error listing services: {e}"
        }


def list_deployments(ns: Optional[str] = None) -> dict:
    """List Kubernetes deployments in a namespace or all namespaces."""
    try:
        if ns:
            deps = [d.metadata.name for d in apps_v1.list_namespaced_deployment(namespace=ns).items]
            return {
                "status": "success",
                "scope": ns,
                "deployments": deps
            }
        else:
            deps = [d.metadata.name for d in apps_v1.list_deployment_for_all_namespaces().items]
            return {
                "status": "success",
                "scope": "all",
                "deployments": deps
            }
    except ApiException as e:
        return {
            "status": "error",
            "message": f"Error listing deployments: {e}"
        }

def list_namespaces() -> dict:
    """List all Kubernetes namespaces."""
    try:
        ns_list = [ns.metadata.name for ns in v1.list_namespace().items]
        return {
            "status": "success",
            "namespaces": ns_list
        }
    except ApiException as e:
        return {
            "status": "error",
            "message": f"Error listing namespaces: {e}"
        }

def get_pod(name: str, namespace: str = "default") -> dict:
    """Get details of a specific Kubernetes pod by name and namespace."""
    try:
        pod = v1.read_namespaced_pod(name=name, namespace=namespace)
        return {
            "status": "success",
            "name": pod.metadata.name,
            "namespace": namespace,
            "pod_status": pod.status.phase,
            "pod_ip": pod.status.pod_ip
        }
    except ApiException as e:
        if e.status == 404:
            return {
                "status": "error",
                "message": f"Pod {name} not found in namespace {namespace}."
            }
        return {
            "status": "error",
            "message": f"Error getting pod {name}: {e}"
        }

    
def list_pods(namespace: Optional[str] = None) -> dict:
    """List all Kubernetes pods in a given namespace or all namespaces if none is specified."""
    try:
        if namespace:
            pods = [pod.metadata.name for pod in v1.list_namespaced_pod(namespace=namespace).items]
            return {"namespace": namespace, "pods": pods}
        else:
            all_pods = v1.list_pod_for_all_namespaces().items
            result = {}
            for pod in all_pods:
                ns = pod.metadata.namespace
                result.setdefault(ns, []).append(pod.metadata.name)
            return {"all_namespaces": result}
    except ApiException as e:
        return {"error": str(e)}


def scale_deployments(name: str, replicas: int, namespace: str = "default") -> dict:
    """Scale a Kubernetes Deployment to a specific number of replicas."""
    try:
        if not isinstance(replicas, int) or replicas < 0:
            return {"status": "error", "msg": "Replicas must be a non-negative integer."}

        patch = {"spec": {"replicas": replicas}}
        apps_v1.patch_namespaced_deployment_scale(name=name, namespace=namespace, body=patch)

        return {
            "status": "success",
            "act": "scale_deploy",
            "name": name,
            "ns": namespace,
            "replicas": replicas,
            "msg": f"{name} scaled to {replicas} in {namespace}"
        }
    except ApiException as e:
        return {
            "status": "error",
            "act": "scale_deploy",
            "name": name,
            "ns": namespace,
            "code": e.status,
            "msg": f"{name} not found" if e.status == 404 else str(e)
        }

def get_deploy_status(name: str, ns: str = "default") -> dict:
    """Get deployment status: replicas and conditions."""
    try:
        dep = apps_v1.read_namespaced_deployment_status(name=name, namespace=ns)
        st = dep.status

        return {
            "status": "success",
            "act": "get_deploy_status",
            "name": name,
            "ns": ns,
            "replicas": {
                "ready": st.ready_replicas or 0,
                "avail": st.available_replicas or 0,
                "updated": st.updated_replicas or 0,
                "total": st.replicas or 0
            },
            "conds": [
                {
                    "type": c.type,
                    "status": c.status,
                    "reason": c.reason,
                    "msg": c.message
                }
                for c in (st.conditions or [])
            ]
        }
    except ApiException as e:
        return {
            "status": "error",
            "act": "get_deploy_status",
            "name": name,
            "ns": ns,
            "code": e.status,
            "msg": f"{name} not found" if e.status == 404 else str(e)
        }

def get_configmap(name: str, ns: str = "default") -> dict:
    """Get ConfigMap details by name and namespace."""
    try:
        cm = v1.read_namespaced_config_map(name=name, namespace=ns)
        return {
            "status": "success",
            "act": "get_cm",
            "name": name,
            "ns": ns,
            "data": cm.data or {}
        }
    except ApiException as e:
        return {
            "status": "error",
            "act": "get_cm",
            "name": name,
            "ns": ns,
            "code": e.status,
            "msg": f"{name} not found" if e.status == 404 else str(e)
        }

def create_secret(name: str, ns: str, data: dict, typ: str = "Opaque") -> dict:
    """Create a Kubernetes Secret."""
    try:
        enc = {k: base64.b64encode(v.encode()).decode() for k, v in data.items()}
        body = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {"name": name},
            "type": typ,
            "data": enc
        }
        v1.create_namespaced_secret(body=body, namespace=ns)
        return {
            "status": "success",
            "act": "create_secret",
            "name": name,
            "ns": ns,
            "type": typ
        }
    except ApiException as e:
        return {
            "status": "error",
            "act": "create_secret",
            "name": name,
            "ns": ns,
            "code": e.status,
            "msg": str(e)
        }

def delete_secret(name: str, ns: str, confirm: bool = False) -> dict:
    """Delete a Kubernetes Secret."""
    try:
        if not confirm:
            return {
                "status": "confirm_required",
                "act": "delete_secret",
                "msg": f"Confirm deletion of secret '{name}' in namespace '{ns}'"
            }
        v1.delete_namespaced_secret(name=name, namespace=ns)
        return {
            "status": "success",
            "act": "delete_secret",
            "name": name,
            "ns": ns
        }
    except ApiException as e:
        return {
            "status": "error",
            "act": "delete_secret",
            "name": name,
            "ns": ns,
            "code": e.status,
            "msg": str(e)
        }

def get_secret(name: str, ns: str, mask: bool = True) -> dict:
    """Fetch a Kubernetes Secret's details. Masks data by default."""
    try:
        sec = v1.read_namespaced_secret(name=name, namespace=ns)
        data = {}
        if sec.data:
            for k, v in sec.data.items():
                val = base64.b64decode(v).decode("utf-8")
                data[k] = "********" if mask else val
        return {
            "status": "success",
            "act": "get_secret",
            "name": name,
            "ns": ns,
            "type": sec.type,
            "data": data or "No data"
        }
    except ApiException as e:
        return {
            "status": "error",
            "act": "get_secret",
            "name": name,
            "ns": ns,
            "code": e.status,
            "msg": str(e)
        }
def create_ingress(name: str, ns: str, host: str, svc: str, port: int, path: str = "/") -> dict:
    """Create a Kubernetes Ingress."""
    try:
        manifest = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {"name": name},
            "spec": {
                "rules": [{
                    "host": host,
                    "http": {
                        "paths": [{
                            "path": path,
                            "pathType": "Prefix",
                            "backend": {
                                "service": {"name": svc, "port": {"number": port}}
                            }
                        }]
                    }
                }]
            }
        }
        networking_v1.create_namespaced_ingress(body=manifest, namespace=ns)
        return {
            "status": "success",
            "act": "create_ingress",
            "name": name,
            "ns": ns,
            "host": host,
            "path": path
        }
    except ApiException as e:
        return {
            "status": "error",
            "act": "create_ingress",
            "name": name,
            "ns": ns,
            "code": e.status,
            "msg": str(e)
        }

def delete_ingress(name: str, ns: str, confirm: bool = False) -> dict:
    """Delete a Kubernetes Ingress."""
    try:
        if not confirm:
            return {
                "status": "confirm_needed",
                "act": "delete_ingress",
                "name": name,
                "ns": ns,
                "msg": f"Confirm deletion by saying: 'yes, delete ingress {name}'"
            }

        networking_v1.delete_namespaced_ingress(name=name, namespace=ns)
        return {
            "status": "success",
            "act": "delete_ingress",
            "name": name,
            "ns": ns
        }

    except ApiException as e:
        return {
            "status": "error",
            "act": "delete_ingress",
            "name": name,
            "ns": ns,
            "code": e.status,
            "msg": str(e)
        }

def get_ingress(name: str, ns: str) -> dict:
    """Get details of a Kubernetes Ingress."""
    try:
        ing = networking_v1.read_namespaced_ingress(name=name, namespace=ns)
        rules = []

        for r in ing.spec.rules or []:
            host = r.host or "N/A"
            paths = []
            for p in r.http.paths or []:
                svc = p.backend.service
                paths.append({
                    "path": p.path or "/",
                    "type": p.path_type or "Prefix",
                    "backend": f"{svc.name}:{svc.port.number}" if svc else "N/A"
                })
            rules.append({"host": host, "paths": paths})

        return {
            "status": "success",
            "act": "get_ingress",
            "name": name,
            "ns": ns,
            "rules": rules or "no_rules"
        }

    except ApiException as e:
        return {
            "status": "not_found" if e.status == 404 else "error",
            "act": "get_ingress",
            "name": name,
            "ns": ns,
            "msg": str(e)
        }


def get_hpa_status(name: str, ns: str = "default") -> str:
    """Get the status of a Horizontal Pod Autoscaler (HPA)."""
    try:
        h = autoscaling_v1.read_namespaced_horizontal_pod_autoscaler_status(name, ns)
        s = f"HPA '{name}' in ns '{ns}':\n"
        s += f"  Ref: {h.spec.scale_target_ref.kind}/{h.spec.scale_target_ref.name}\n"
        s += f"  Min: {h.spec.min_replicas}, Max: {h.spec.max_replicas}\n"

        if h.status:
            s += f"  Cur: {h.status.current_replicas}, Des: {h.status.desired_replicas}\n"
            if h.status.current_metrics:
                s += "  Metrics:\n"
                for i, m in enumerate(h.status.current_metrics):
                    if m.resource:
                        cur = m.resource.current.average_value or m.resource.current.average_utilization
                        tgt = h.spec.metrics[i].resource.target.average_value or h.spec.metrics[i].resource.target.average_utilization
                        s += f"    - {m.resource.name}: {cur} (Tgt: {tgt})\n"
                    elif m.external:
                        cur = m.external.current.average_value
                        tgt = h.spec.metrics[i].external.target.average_value
                        s += f"    - {m.external.metric_name}: {cur} (Tgt: {tgt})\n"
                    elif m.pods:
                        cur = m.pods.current.average_value
                        tgt = h.spec.metrics[i].pods.target.average_value
                        s += f"    - {m.pods.metric_name}: {cur} (Tgt: {tgt})\n"
            if h.status.conditions:
                s += "  Cond:\n"
                for c in h.status.conditions:
                    s += f"    - {c.type}: {c.status} (R: {c.reason}, M: {c.message})\n"
        return s
    except ApiException as e:
        if e.status == 404:
            return f"HPA '{name}' not found in ns '{ns}'."
        return f"Err getting HPA '{name}': {e}"


def create_pvc(n: str, ns: str, sc: str, sz: str, am: list = ['ReadWriteOnce']) -> str:
    """Create a PVC."""
    try:
        pvc = {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {"name": n},
            "spec": {
                "accessModes": am,
                "storageClassName": sc,
                "resources": {"requests": {"storage": sz}}
            }
        }
        v1.create_namespaced_persistent_volume_claim(body=pvc, namespace=ns)
        return f"PVC '{n}' created in ns '{ns}' ({sz})."
    except ApiException as e:
        return f"Err creating PVC '{n}': {e}"

def del_pvc(n: str, ns: str, c: bool = False) -> str:
    """Delete a PVC."""
    try:
        if not c:
            return f"Confirm to delete PVC '{n}' in ns '{ns}'. Say: yes, delete pvc {n}."
        v1.delete_namespaced_persistent_volume_claim(name=n, namespace=ns)
        return f"PVC '{n}' deleted from ns '{ns}'."
    except ApiException as e:
        if e.status == 404:
            return f"PVC '{n}' not found in ns '{ns}'."
        return f"Err deleting PVC '{n}': {e}"

def get_pvc(n: str, ns: str) -> str:
    """Get PVC info."""
    try:
        p = v1.read_namespaced_persistent_volume_claim(name=n, namespace=ns)
        s = p.status.phase
        c = p.status.capacity.get('storage', 'N/A') if p.status.capacity else 'N/A'
        v = p.spec.volume_name or 'Unbound'
        sc = p.spec.storage_class_name
        return (f"PVC '{n}' in ns '{ns}':\n"
                f"  Status: {s}\n"
                f"  Vol: {v}\n"
                f"  Cap: {c}\n"
                f"  SC: {sc}\n"
                f"  AM: {', '.join(p.spec.access_modes)}")
    except ApiException as e:
        if e.status == 404:
            return f"PVC '{n}' not found in ns '{ns}'."
        return f"Err getting PVC '{n}': {e}"

def list_pvcs(ns: Optional[str] = None) -> str:
    """List PVCs."""
    try:
        if ns:
            p = [x.metadata.name for x in v1.list_namespaced_persistent_volume_claim(namespace=ns).items]
            return f"PVCs in ns {ns}: {', '.join(p) if p else 'None'}"
        p = [x.metadata.name for x in v1.list_persistent_volume_claim_for_all_namespaces().items]
        return f"All PVCs: {', '.join(p) if p else 'None'}"
    except ApiException as e:
        return f"Err listing PVCs: {e}"

def create_hpa(name: str, ns: str, scale_target_kind: str, scale_target_name: str, min_replicas: int, max_replicas: int, cpu_utilization: int) -> str:
    """Create a HorizontalPodAutoscaler targeting CPU utilization."""
    try:
        hpa_manifest = {
            "apiVersion": "autoscaling/v1",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {"name": name},
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": scale_target_kind,
                    "name": scale_target_name
                },
                "minReplicas": min_replicas,
                "maxReplicas": max_replicas,
                "targetCPUUtilizationPercentage": cpu_utilization
            }
        }
        autoscaling_v1.create_namespaced_horizontal_pod_autoscaler(body=hpa_manifest, namespace=ns)
        return f"HPA '{name}' created successfully in namespace '{ns}'."
    except ApiException as e:
        return f"Error creating HPA '{name}': {e}"

def delete_hpa(name: str, ns: str, confirm: bool = False) -> str:
    """Delete a HorizontalPodAutoscaler by name and namespace."""
    try:
        if not confirm:
            return f"Confirmation required to delete HPA '{name}' in namespace '{ns}'. Please confirm by saying 'yes, delete hpa {name}'."
        autoscaling_v1.delete_namespaced_horizontal_pod_autoscaler(name=name, namespace=ns)
        return f"HPA '{name}' deleted successfully from namespace '{ns}'."
    except ApiException as e:
        if e.status == 404:
            return f"HPA '{name}' not found in namespace '{ns}'."
        return f"Error deleting HPA '{name}': {e}"

def list_hpas(namespace: Optional[str] = None) -> str:
    """List all HorizontalPodAutoscalers in a given namespace or all namespaces."""
    try:
        if namespace:
            hpas = [hpa.metadata.name for hpa in autoscaling_v1.list_namespaced_horizontal_pod_autoscaler(namespace=namespace).items]
            return f"HPAs in namespace {namespace}: {', '.join(hpas) if hpas else 'None'}"
        else:
            hpas = [hpa.metadata.name for hpa in autoscaling_v1.list_horizontal_pod_autoscaler_for_all_namespaces().items]
            return f"All HPAs: {', '.join(hpas) if hpas else 'None'}"
    except ApiException as e:
        return f"Error listing HPAs: {e}"

def list_network_policies(namespace: str) -> str:
    """List all NetworkPolicies in a given namespace."""
    try:
        policies = [p.metadata.name for p in networking_v1.list_namespaced_network_policy(namespace=namespace).items]
        return f"NetworkPolicies in namespace {namespace}: {', '.join(policies) if policies else 'None'}"
    except ApiException as e:
        return f"Error listing NetworkPolicies: {e}"

resource_api_map = {
    "pod": (v1, "read_namespaced_pod"),
    "deployment": (apps_v1, "read_namespaced_deployment"),
    "service": (v1, "read_namespaced_service"),
    "namespace": (v1, "read_namespace"), # Note: no 'namespaced' for namespace itself
    "node": (v1, "read_node"), # Note: no 'namespaced' for node itself
    "configmap": (v1, "read_namespaced_config_map"),
    "secret": (v1, "read_namespaced_secret"),
    "ingress": (networking_v1, "read_namespaced_ingress"),
    "pvc": (v1, "read_namespaced_persistent_volume_claim"),
    "hpa": (autoscaling_v1, "read_namespaced_horizontal_pod_autoscaler"),
    "networkpolicy": (networking_v1, "read_namespaced_network_policy"),
    # Add more as needed
}

def desc_res(rtype: str, name: str, ns: Optional[str] = None) -> str:
    """Mimic 'kubectl describe' for K8s resources."""
    rtype = rtype.lower()
    if rtype not in resource_api_map:
        return f"Unsupported: {rtype}. Use: {', '.join(resource_api_map.keys())}"

    api, method = resource_api_map[rtype]
    try:
        if "namespaced" in method:
            if not ns:
                return f"Namespace required for '{rtype}'."
            res = getattr(api, method)(name=name, namespace=ns)
        else:
            res = getattr(api, method)(name=name)

        return f"{rtype} '{name}' in ns '{ns or 'N/A'}':\n" + yaml.dump(res.to_dict(), default_flow_style=False)
    except ApiException as e:
        if e.status == 404:
            return f"{rtype} '{name}' not found in ns '{ns or 'N/A'}'."
        return f"Err: {rtype} '{name}': {e}"
    except Exception as e:
        return f"Unexpected err: {rtype} '{name}': {e}"

def get_node_status(name: str) -> str:
    """Get K8s node status."""
    try:
        n = v1.read_node_status(name=name)
        out = f"Node '{name}':\n"

        conds = "No conditions."
        if n.status.conditions:
            conds = "\n".join([f"- {c.type}: {c.status} ({c.reason}, {c.message})" for c in n.status.conditions])
        out += f"  Conds:\n{conds}\n"

        addrs = "No addrs."
        if n.status.addresses:
            addrs = "\n".join([f"- {a.type}: {a.address}" for a in n.status.addresses])
        out += f"  Addrs:\n{addrs}\n"

        if n.status.capacity:
            out += f"  Cap: {n.status.capacity}\n"
        if n.status.allocatable:
            out += f"  Alloc: {n.status.allocatable}\n"

        return out
    except ApiException as e:
        return f"Node '{name}' not found." if e.status == 404 else f"Err node '{name}': {e}"

def get_cluster_status() -> str:
    """Get K8s node status."""
    try:
        nodes = v1.list_node().items
        total = len(nodes)
        ready = 0
        summary = []

        for n in nodes:
            name = n.metadata.name
            cond = next((c for c in n.status.conditions if c.type == "Ready"), None)
            if cond and cond.status == "True":
                ready += 1
                summary.append(f"- {name}: Ready")
            else:
                st = cond.status if cond else "Unknown"
                rsn = cond.reason if cond else "N/A"
                summary.append(f"- {name}: Not Ready ({st}, {rsn})")

        return f"K8s Cluster:\n  Nodes: {total}\n  Ready: {ready}\nDetails:\n" + "\n".join(summary)
    except ApiException as e:
        return f"Err getting cluster: {e}"

def get_events(ns: Optional[str] = None, selector: Optional[str] = None, limit: int = 50) -> str:
    """Get K8s events (filtered by ns/selector)."""
    try:
        if ns:
            evs = v1.list_namespaced_event(namespace=ns, field_selector=selector, limit=limit).items
        else:
            evs = v1.list_event_for_all_namespaces(field_selector=selector, limit=limit).items

        if not evs:
            scope = f"ns '{ns}'" if ns else "all ns"
            filt = f" with selector '{selector}'" if selector else ""
            return f"No events in {scope}{filt}."

        out = [f"[{e.last_timestamp or e.event_time}] {e.type} {e.reason} ({e.involved_object.kind}/{e.involved_object.name}): {e.message}" for e in evs]
        return "K8s Events:\n" + "\n".join(out)
    except ApiException as e:
        return f"Err fetching events: {e}"

def get_pod_logs(name: str, ns: str = "default", lines: int = 50) -> str:
    """Get last N lines of pod logs."""
    try:
        log = v1.read_namespaced_pod_log(name=name, namespace=ns, tail_lines=lines)
        if not log.strip():
            return f"No logs for pod '{name}' in ns '{ns}'."
        return f"Last {lines} log lines for pod '{name}' in ns '{ns}':\n{log}"
    except ApiException as e:
        if e.status == 404:
            return f"Pod '{name}' not found in ns '{ns}'."
        return f"Err getting logs for pod '{name}': {e}"


# A list of all available tools for easy import.
k8s_tools = [
    list_namespaces, get_pod, list_pods, get_pod_logs,
    get_events, get_deploy_status, get_configmap,
    get_secret, create_secret, delete_secret,
    get_ingress, create_ingress, delete_ingress,
    get_node_status, get_cluster_status,
    get_hpa_status, list_hpas, create_hpa, delete_hpa,
    desc_res,
    create_pod, del_pod,
    create_ns, del_ns,
    create_deploy,delete_deploy, scale_deployments,
    create_svc, delete_svc,
    list_svc, list_deployments,
    # PVC Tools
    create_pvc, del_pvc, get_pvc, list_pvcs,
    # Networking Tools
    list_network_policies,
]



